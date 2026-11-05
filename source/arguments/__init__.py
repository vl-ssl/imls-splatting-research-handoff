from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        super().__init__(parser, "Pipeline Parameters")
     
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations      = 20_000
        self.grid_resolution = 64
        self.SSAA            = 1
        self.meshscale       = 3.0
        self.auto_mesh_domain = False
        self.geometry_only = False
        self.lambda_silhouette_pyramid = 0.0
        self.lambda_silhouette_iou = 0.0
        self.lambda_silhouette_recall = 0.0
        self.lambda_silhouette_boundary = 0.0
        self.lambda_visible_contour = 0.0
        self.visible_contour_samples = 2048
        self.lambda_opaque_boundary_region = 0.0
        self.lambda_opaque_boundary_edges = 0.0
        self.opaque_boundary_samples = 2048
        self.opaque_boundary_holdout_stride = 0
        self.opaque_boundary_max_displacement_voxels = 0.0
        self.opaque_boundary_audit_gradient = False
        self.lambda_surface_smooth = 0.0
        self.lambda_surface_tangent = 0.0
        self.lambda_gt_normal_smooth = 0.0
        self.lambda_gt_normal_gradient = 0.0
        self.lambda_point_anchor = 0.0
        self.lambda_point_tangent = 0.0
        self.lambda_image    = 0.2
        self.lambda_normal   = 0.005
        # V11: focus a separate geometry-only term on the worst normal
        # residuals inside GT-smooth opaque regions. Defaults keep every
        # historical version bit-for-bit on its previous loss path.
        self.lambda_normal_cvar = 0.0
        self.normal_cvar_fraction = 0.2
        self.normal_cvar_cosine = 0.995
        self.normal_cvar_erode = 5
        self.lambda_depth    = 0.0
        self.lambda_disparity = 0.0
        self.lambda_mask     = 0.1
        self.lambda_lpips    = 0.0
        self.lpips_interval  = 4
        self.lpips_resolution = 256
        self.lpips_standard_input = False
        self.reflection_grad_scale = 0.05
        # V22-R: roughness-conditioned directional factorization for opaque
        # specular surfaces. Disabled by default for historical checkpoints.
        self.factorized_reflection = False
        self.lambda_reflection_tv = 0.0
        self.lambda_reflection_normal = 0.0
        # V31: propagate reflection-only micro normals across regions that are
        # geometrically continuous according to training GT normals/masks.
        self.lambda_reflection_normal_tv = 0.0
        self.reflection_normal_tv_cosine = 0.995
        self.reflection_normal_tv_erode = 3
        # -1 keeps the legacy iteration schedule. Values in [0, 1] select a
        # fixed mesh/SDF normal mixture for reflection rendering.
        self.mesh_normal_weight = -1.0
        self.implicit_pixel_normal_weight = 0.0
        # V4 surface repair.  All defaults are no-ops so old checkpoints keep
        # their exact rendering behaviour unless a V4 run enables them.
        self.sdf_close_weight = 0.0
        self.sdf_smooth_weight = 0.0
        self.sdf_smooth_iterations = 1
        # V12: narrow-band, normal-aware Taubin diffusion of the implicit
        # field. Zero weight preserves every historical surface exactly.
        self.sdf_taubin_weight = 0.0
        self.sdf_taubin_iterations = 0
        self.sdf_taubin_cosine = 0.95
        self.sdf_taubin_band_voxels = 2.0
        self.sdf_taubin_mu = -0.53
        self.coverage_hole_fill_min_neighbors = 0
        self.coverage_hole_fill_iterations = 1
        # V19: confidence-aware IMLS field regularization.  All zero defaults
        # preserve prior checkpoints and scripts exactly.
        self.imls_confidence_smooth_weight = 0.0
        self.imls_confidence_smooth_iterations = 0
        self.imls_confidence_floor = 0.35
        self.imls_confidence_band_voxels = 2.0
        # V21: optional original-support gate for visual-hull auxiliary
        # points. Zero preserves the V20 weighted-moment path.
        self.imls_auxiliary_primary_support_floor = 0.0
        # V29: local bidirectional consistency between explicit oriented
        # points and the implicit IMLS zero level set. All defaults are no-ops
        # so archived versions keep their exact training path.
        self.lambda_imls_surface = 0.0
        self.lambda_imls_sign = 0.0
        self.lambda_imls_tangent = 0.0
        self.imls_consistency_interval = 1
        self.imls_consistency_samples = 8192
        self.imls_consistency_offset_voxels = 1.0
        self.imls_consistency_support_floor = 1e-4
        # V30: an independently optimized, low-resolution continuous SDF
        # residual. It changes topology through the implicit field without
        # asking potentially unoriented source-point normals to define inside.
        self.implicit_sdf_grid_resolution = 0
        self.implicit_sdf_amplitude_voxels = 0.0
        self.lambda_implicit_sdf_magnitude = 0.0
        self.lambda_implicit_sdf_tv = 0.0
        self.mesh_component_min_faces = 0
        self.mesh_component_min_fraction = 0.0
        self.normal_smooth_weight = 0.0
        self.normal_smooth_iterations = 0
        self.normal_smooth_cosine = 0.5
        self.mesh_vertex_smooth_weight = 0.0
        self.mesh_vertex_smooth_iterations = 0
        self.mesh_vertex_smooth_cosine = 0.9
        self.mesh_vertex_smooth_taubin = -0.53
        self.mesh_vertex_boundary_weight = 0.1
        self.save_final_geometry_maps = 1
        self.save_final_depth_maps = 1
        # Fast ablations may skip the train split.  Final reproducibility runs
        # keep the default and therefore still export every available view.
        self.final_render_train = 1
        self.final_render_test_limit = 0
        # Save the final point/checkpoint without exporting any views. Useful
        # for an intermediate geometry stage followed by a separate evaluator.
        self.skip_final_render = False
        # V20 diagnostics can stop after the initial IMLS extraction, avoiding
        # all-view image export when only kernel-support statistics are needed.
        self.support_diagnostics_only = False
        self.depth_model_path = "./Depth-Anything-V2-Small-hf"

        self.upsample_from     = 0
        self.upsample_interval = 4000
        self.upsample_until    = 13_000
        self.resample_from     = 2000
        self.resample_interval = 1000
        self.resample_until    = 15_000
        
        self.position_lr_init       = 0.00005
        self.position_lr_final      = 0.00005
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps  = 20_000
        self.normal_lr    = 0.005
        self.scaling_lr   = 0.005
        self.feature_lr   = 0.025
        self.rendernet_lr = 0.001
        # V3 can fine-tune only the zero-initialized reflection residual while
        # keeping the checkpoint-compatible renderer exactly fixed.
        self.residual_lr = 0.001
        self.position_residual_lr = 0.001
        self.environment_lr = 0.001
        self.reflection_map_lr = 0.001
        self.reflection_detail_lr = 0.0
        self.reflection_normal_lr = 0.0
        self.reflection_factor_lr = 0.0
        self.implicit_sdf_lr = 0.0
        self.freeze_base_rendernet = False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
