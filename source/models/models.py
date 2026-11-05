import numpy as np
import torch
import torch.nn.functional as F


def positional_encoding(positions, freqs):
    bands = 2 ** torch.arange(freqs, device=positions.device, dtype=positions.dtype)
    phase = (positions[..., None] * bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1],)
    )
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class rendernet(torch.nn.Module):
    """Checkpoint-compatible IMLS renderer with a safe residual reflector."""

    def __init__(
        self,
        num_theta=8,
        num_phi=16,
        dim_feature=192,
        featureC=256,
        device="cpu",
        view_pe=-1,
        fea_pe=-1,
        btn_freq=None,
        reflection_grad_scale=0.0,
        factorized_reflection=False,
    ):
        super().__init__()
        del num_theta, num_phi
        self.ch_cd = 3
        self.ch_s = 3
        self.ch_bottleneck = 128
        self.ch_normal_dot_viewdir = 1
        self.view_pe = view_pe
        self.fea_pe = fea_pe
        self.reflection_grad_scale = reflection_grad_scale
        self.factorized_reflection = bool(factorized_reflection)

        if btn_freq is not None and len(btn_freq) >= 2:
            frequencies = torch.linspace(
                np.log2(btn_freq[0]), np.log2(btn_freq[1]), self.ch_bottleneck
            )
            frequencies = torch.exp2(frequencies)
        else:
            frequencies = None
        self.register_buffer("btn_freq", frequencies)

        spatial_in = dim_feature
        if fea_pe > 0:
            spatial_in += dim_feature * fea_pe * 2
        self.spatial_mlp = torch.nn.Sequential(
            torch.nn.Linear(spatial_in, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, self.ch_cd + self.ch_s + self.ch_bottleneck),
        )

        directional_in = self.ch_bottleneck + self.ch_normal_dot_viewdir
        if view_pe > -1:
            directional_in += 6
        if view_pe > 0:
            directional_in += 6 * view_pe * 2
        self.directional_mlp = torch.nn.Sequential(
            torch.nn.Linear(directional_in, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, 3),
        )

        # Old *_op checkpoints do not contain this branch. Zero initialization
        # makes their initial RGB exactly reproducible before v3 fine-tuning.
        self.residual_mlp = torch.nn.Sequential(
            torch.nn.Linear(directional_in, featureC // 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC // 2, 3),
        )
        torch.nn.init.zeros_(self.residual_mlp[-1].weight)
        torch.nn.init.zeros_(self.residual_mlp[-1].bias)
        self.residual_scale = 0.15

        # A specular material boundary is fixed on the opaque surface, not in
        # camera space.  This zero-initialized branch exposes the actual surface
        # coordinate so that bands/coatings can be represented without moving
        # the reconstructed geometry or invalidating older checkpoints.
        self.position_pe = 4
        position_residual_in = directional_in + 3 + 3 * self.position_pe * 2
        self.position_residual_mlp = torch.nn.Sequential(
            torch.nn.Linear(position_residual_in, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC // 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC // 2, 3),
        )
        torch.nn.init.zeros_(self.position_residual_mlp[-1].weight)
        torch.nn.init.zeros_(self.position_residual_mlp[-1].bias)
        self.position_residual_scale = 0.5

        # Global reflected illumination is shared by all opaque surface points.
        # A dedicated high-frequency direction branch avoids forcing sharp
        # environment boundaries through the per-point material bottleneck.
        self.environment_pe = 6
        environment_in = 3 + 3 * self.environment_pe * 2 + 1
        self.environment_mlp = torch.nn.Sequential(
            torch.nn.Linear(environment_in, featureC * 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC * 2, featureC * 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC * 2, 3),
        )
        torch.nn.init.zeros_(self.environment_mlp[-1].weight)
        torch.nn.init.zeros_(self.environment_mlp[-1].bias)
        self.environment_scale = 0.75
        # V5: material-conditioned directional factorization, inspired by the
        # concept in Ref-GS; independent implementation (no third-party code).
        # A periodic angular grid represents sharp lighting without squeezing
        # it through a small Fourier MLP. The zero readout preserves V4 RGB.
        self.directional_texture = torch.nn.Parameter(torch.randn(1, 12, 64, 128) * 0.01)
        self.directional_material = torch.nn.Linear(dim_feature + 1, 36)
        torch.nn.init.zeros_(self.directional_material.weight)
        torch.nn.init.zeros_(self.directional_material.bias)

        # V6: directly sampled RGB reflection map with a surface-conditioned
        # gate.  Unlike the latent factorization above, its zero-valued map
        # receives a useful gradient on the first optimization step.  The map
        # is initialized to zero so older checkpoints render identically.
        self.reflection_texture = torch.nn.Parameter(
            torch.zeros(1, 3, 128, 256)
        )
        self.reflection_gate = torch.nn.Linear(dim_feature + 1, 1)
        torch.nn.init.zeros_(self.reflection_gate.weight)
        torch.nn.init.constant_(self.reflection_gate.bias, -2.0)
        self.reflection_texture_scale = 0.75
        # V13: zero-initialized high-frequency residual environment map.
        # It sharpens reflected contours without changing any loaded V10 RGB
        # before explicit detail-only fine-tuning.
        self.reflection_detail_texture = torch.nn.Parameter(
            torch.zeros(1, 3, 256, 512)
        )
        self.reflection_detail_scale = 0.35

        # V14: a view-independent micro-normal correction used only to look up
        # reflected radiance.  Geometry normals and exported normal maps stay
        # untouched.  The zero final layer makes every older checkpoint render
        # exactly as before until this branch is explicitly optimized.
        self.reflection_normal_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_feature + 3 + 3 * self.position_pe * 2, featureC),
            torch.nn.GELU(),
            torch.nn.Linear(featureC, featureC // 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC // 2, 3),
        )
        torch.nn.init.zeros_(self.reflection_normal_mlp[-1].weight)
        torch.nn.init.zeros_(self.reflection_normal_mlp[-1].bias)
        self.reflection_normal_scale = 0.2

        # V22-R: independent implementation of roughness-aware directional
        # factorization, conceptually inspired by Ref-GS/Ref-NeRF. The zero
        # material readout preserves old-checkpoint RGB before optimization.
        self.reflection_factor_texture = torch.nn.Parameter(
            torch.randn(1, 8, 128, 256) * 0.01
        )
        self.reflection_factor_material = torch.nn.Linear(dim_feature, 3 * 8)
        torch.nn.init.zeros_(self.reflection_factor_material.weight)
        torch.nn.init.zeros_(self.reflection_factor_material.bias)
        self.reflection_factor_gate = torch.nn.Linear(dim_feature, 1)
        torch.nn.init.zeros_(self.reflection_factor_gate.weight)
        torch.nn.init.constant_(self.reflection_factor_gate.bias, -1.5)
        self.reflection_roughness_mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_feature, featureC // 2),
            torch.nn.GELU(),
            torch.nn.Linear(featureC // 2, 1),
        )
        torch.nn.init.zeros_(self.reflection_roughness_mlp[-1].weight)
        torch.nn.init.constant_(self.reflection_roughness_mlp[-1].bias, -4.0)
        self.reflection_factor_scale = 0.5
        self.to(device)

    @staticmethod
    def reflect(viewdir, normal):
        return 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir

    def spatial_mlp_forward(self, x):
        sections = [self.ch_cd, self.ch_s, self.ch_bottleneck]
        return torch.split(self.spatial_mlp(x), sections, dim=-1)

    def sample_reflection_mip(self, refdir, roughness):
        """Sample a periodic spherical feature pyramid at continuous LOD.

        Average-pool mips give a controlled directional bandwidth; this is not
        claimed to be a physical GGX prefilter.
        """
        longitude = torch.atan2(refdir[:, 1], refdir[:, 0]) / np.pi
        latitude = 2 * torch.acos(
            refdir[:, 2].clamp(-1 + 1e-6, 1 - 1e-6)
        ) / np.pi - 1
        pyramid = [self.reflection_factor_texture]
        while len(pyramid) < 7 and min(pyramid[-1].shape[-2:]) > 2:
            pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))
        lod = roughness.reshape(-1) * (len(pyramid) - 1)
        result = refdir.new_zeros((refdir.shape[0], pyramid[0].shape[1]))
        for level, texture_level in enumerate(pyramid):
            padded = F.pad(texture_level, (1, 1, 0, 0), mode="circular")
            width = texture_level.shape[-1]
            uv = torch.stack(
                (longitude * width / (width + 2), latitude), dim=-1
            )[None, None]
            sampled = F.grid_sample(
                padded, uv, mode="bilinear", padding_mode="border",
                align_corners=False,
            )[0, :, 0].T
            weight = (1.0 - (lod - level).abs()).clamp(0.0, 1.0)[:, None]
            result = result + sampled * weight
        return result

    def forward(
        self,
        viewdir_all,
        normal_all,
        feature_all,
        alpha,
        gb_pos_all=None,
        camera_center=None,
    ):
        del camera_center
        batch, height, width = viewdir_all.shape[:3]
        mask = alpha.reshape(-1) > 0
        if not mask.any():
            return {"rgb_map": viewdir_all.new_zeros((batch, height, width, 3))}

        viewdir = F.normalize(viewdir_all.reshape(-1, 3)[mask], dim=-1, eps=1e-6)
        normal = F.normalize(normal_all.reshape(-1, 3)[mask], dim=-1, eps=1e-6)
        feature = feature_all.reshape(-1, feature_all.shape[-1])[mask]
        if gb_pos_all is None:
            surface_position = feature.new_zeros((feature.shape[0], 3))
        else:
            surface_position = gb_pos_all.reshape(-1, 3)[mask]

        scale = self.reflection_grad_scale
        viewdir = viewdir.detach() + scale * (viewdir - viewdir.detach())
        normal = normal.detach() + scale * (normal - normal.detach())

        spatial_input = [feature]
        if self.fea_pe > 0:
            spatial_input.append(positional_encoding(feature, self.fea_pe))
        diffuse_raw, tint, bottleneck = self.spatial_mlp_forward(
            torch.cat(spatial_input, dim=-1)
        )
        reflection_normal_input = torch.cat(
            (
                feature,
                surface_position,
                positional_encoding(surface_position, self.position_pe),
            ),
            dim=-1,
        )
        reflection_normal_offset = self.reflection_normal_scale * torch.tanh(
            self.reflection_normal_mlp(reflection_normal_input)
        )
        # Keep the learned correction tangent to the reconstructed opaque
        # surface. This changes reflection direction without allowing the
        # appearance loss to invent a second geometric surface.
        reflection_normal_offset = reflection_normal_offset - (
            reflection_normal_offset * normal
        ).sum(dim=-1, keepdim=True) * normal
        specular_normal = F.normalize(
            normal + reflection_normal_offset, dim=-1, eps=1e-6
        )
        refdir = F.normalize(
            self.reflect(-viewdir, specular_normal), dim=-1, eps=1e-6
        )
        if self.btn_freq is not None:
            bottleneck = bottleneck + torch.sin(bottleneck * self.btn_freq[None])

        n_dot_v = ((-viewdir) * normal).sum(dim=-1, keepdim=True).clamp(-1, 1)
        directional_input = [bottleneck, n_dot_v]
        if self.view_pe > -1:
            directional_input.append(viewdir)
        if self.view_pe > 0:
            directional_input.append(positional_encoding(viewdir, self.view_pe))
        if self.view_pe > -1:
            directional_input.append(refdir)
        if self.view_pe > 0:
            directional_input.append(positional_encoding(refdir, self.view_pe))
        directional_input = torch.cat(directional_input, dim=-1)

        base_raw = diffuse_raw + tint * self.directional_mlp(directional_input)
        residual_raw = self.residual_scale * torch.tanh(
            self.residual_mlp(directional_input)
        )
        position_input = torch.cat(
            (
                directional_input,
                surface_position,
                positional_encoding(surface_position, self.position_pe),
            ),
            dim=-1,
        )
        position_residual_raw = self.position_residual_scale * torch.tanh(
            self.position_residual_mlp(position_input)
        )
        environment_input = torch.cat(
            (
                refdir,
                positional_encoding(refdir, self.environment_pe),
                n_dot_v,
            ),
            dim=-1,
        )
        environment_raw = self.environment_scale * torch.tanh(
            self.environment_mlp(environment_input)
        )
        longitude = torch.atan2(refdir[:, 1], refdir[:, 0]) / np.pi
        latitude = 2 * torch.acos(refdir[:, 2].clamp(-1+1e-6, 1-1e-6)) / np.pi - 1
        texture = F.pad(self.directional_texture, (1, 1, 0, 0), mode='circular')
        # align_corners=False: periodic seam sits halfway between edge texels.
        uv = torch.stack((longitude * 128 / 130, latitude), -1)[None, None]
        angular_features = F.grid_sample(texture, uv, mode='bilinear', padding_mode='border',
                                         align_corners=False)[0, :, 0].T
        coefficients = self.directional_material(torch.cat((feature, n_dot_v), -1)).reshape(-1, 3, 12)
        directional_raw = (coefficients * angular_features[:, None]).sum(-1)
        reflection_texture = F.pad(
            self.reflection_texture, (1, 1, 0, 0), mode='circular'
        )
        reflection_width = self.reflection_texture.shape[-1]
        reflection_uv = torch.stack(
            (
                longitude * reflection_width / (reflection_width + 2),
                latitude,
            ),
            -1,
        )[None, None]
        reflected_rgb = F.grid_sample(
            reflection_texture,
            reflection_uv,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )[0, :, 0].T
        reflection_detail_texture = F.pad(
            self.reflection_detail_texture, (1, 1, 0, 0), mode='circular'
        )
        detail_width = self.reflection_detail_texture.shape[-1]
        detail_uv = torch.stack(
            (
                longitude * detail_width / (detail_width + 2),
                latitude,
            ),
            -1,
        )[None, None]
        reflected_detail_rgb = F.grid_sample(
            reflection_detail_texture,
            detail_uv,
            mode='bilinear',
            padding_mode='border',
            align_corners=False,
        )[0, :, 0].T
        reflection_gate = torch.sigmoid(
            self.reflection_gate(torch.cat((feature, n_dot_v), -1))
        )
        reflection_raw = (
            reflection_gate
            * (
                self.reflection_texture_scale * torch.tanh(reflected_rgb)
                + self.reflection_detail_scale * torch.tanh(reflected_detail_rgb)
            )
        )
        if self.factorized_reflection:
            roughness = torch.sigmoid(self.reflection_roughness_mlp(feature))
            mip_features = self.sample_reflection_mip(refdir, roughness)
            material = self.reflection_factor_material(feature).reshape(-1, 3, 8)
            factor_gate = torch.sigmoid(self.reflection_factor_gate(feature))
            factor_raw = self.reflection_factor_scale * factor_gate * torch.tanh(
                (material * mip_features[:, None]).sum(dim=-1)
            )
        else:
            roughness = feature.new_zeros((feature.shape[0], 1))
            factor_raw = feature.new_zeros((feature.shape[0], 3))
        rgb = torch.sigmoid(
            base_raw + residual_raw + position_residual_raw + environment_raw
            + directional_raw + reflection_raw + factor_raw
        )

        def scatter(values):
            result = values.new_zeros((batch * height * width, values.shape[-1]))
            result[mask] = values
            return result.reshape(batch, height, width, values.shape[-1])

        diffuse = torch.sigmoid(diffuse_raw)
        return {
            "rgb_map": scatter(rgb),
            "diffuse_map": scatter(diffuse),
            "specular_map": scatter(rgb - diffuse),
            "reflection_normal_offset_map": scatter(reflection_normal_offset),
            "roughness_map": scatter(roughness),
            "factorized_reflection_map": scatter(factor_raw),
        }
