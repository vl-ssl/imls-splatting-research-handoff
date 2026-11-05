#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t CHANNELS>
__global__ void
splatCUDA(
	int num_computed,
	int num_points,
	int resolution,
	const float* __restrict__ means3D,
	const float* __restrict__ normals,
	const float* __restrict__ scales,
	const float* __restrict__ features,
	const uint2* __restrict__ phases_ranges,
	const uint32_t* __restrict__ phases_tiles,
	const uint32_t* __restrict__ point_list,
	const uint32_t* __restrict__ tiles_list,
	const int* __restrict__ grad_flags,
	const float* __restrict__ out_sums,
	const float* __restrict__ out_sdfs,
	const float* __restrict__ out_feat,
	const float* __restrict__ dL_dout_sums,
	const float* __restrict__ dL_dout_sdfs,
	const float* __restrict__ dL_dout_feat,
	float* __restrict__ dL_dmeans3D,
	float* __restrict__ dL_dnormals,
	float* __restrict__ dL_dscales,
	float* __restrict__ dL_dfeatures)
{
	// Identify current tile and associated min/max pixel range.
	auto idx = cg::this_grid().thread_rank();

	if (idx >= num_computed)
		return;
	
	int    point_id = point_list[idx];
	float3 point    = {means3D[point_id*3], means3D[point_id*3+1], means3D[point_id*3+2]};
	float3 normal   = {normals[point_id*3], normals[point_id*3+1], normals[point_id*3+2]};
	float  scale    = scales[point_id];

	uint32_t tile_id  = tiles_list[idx];

	uint32_t num_blocks_x = (resolution + BLOCK_X - 1) / BLOCK_X;
	uint32_t num_blocks_y = (resolution + BLOCK_Y - 1) / BLOCK_Y;

	int block_idz = (int)( tile_id / (num_blocks_y*num_blocks_x));
	int block_idy = (int)((tile_id - block_idz*(num_blocks_y*num_blocks_x)) / num_blocks_x);
	int block_idx = (int)( tile_id - block_idz*(num_blocks_y*num_blocks_x) - block_idy*num_blocks_x);

	uint3 cell_id_min = {block_idx * BLOCK_X, block_idy * BLOCK_Y, block_idz * BLOCK_Z};
	uint3 cell_id_max = {min(cell_id_min.x+BLOCK_X, resolution), min(cell_id_min.y+BLOCK_Y, resolution), min(cell_id_min.z+BLOCK_Z, resolution) };

	float dL_dp_i_x = 0.0f;
	float dL_dp_i_y = 0.0f;
	float dL_dp_i_z = 0.0f;
	float dL_dn_i_x = 0.0f;
	float dL_dn_i_y = 0.0f;
	float dL_dn_i_z = 0.0f;
	float dL_ds_i   = 0.0f;
	float dL_da_i   = 0.0f;
	float dL_dfeature_i[CHANNELS] = { 0 };

	float feat[CHANNELS]     = { 0 };
	float dL_dfeat[CHANNELS] = { 0 };

	int all_valid_flag = 0;
	for (int cell_id_x = cell_id_min.x; cell_id_x < cell_id_max.x; cell_id_x++)
	{
		for (int cell_id_y = cell_id_min.y; cell_id_y < cell_id_max.y; cell_id_y++)
		{	
			for (int cell_id_z = cell_id_min.z; cell_id_z < cell_id_max.z; cell_id_z++)
			{
				uint32_t cell_id_flat = cell_id_z * resolution * resolution + cell_id_y * resolution + cell_id_x;

				if (grad_flags[cell_id_flat] == 1)
				{
					all_valid_flag = 1;

					float3 cell = {(float)cell_id_x / (resolution-1), (float)cell_id_y / (resolution-1), (float)cell_id_z / (resolution-1)};

					// get cell info.
					float sum = out_sums[cell_id_flat];
					float sdf = out_sdfs[cell_id_flat];
					for (int ch = 0; ch < CHANNELS; ch++)
						feat[ch] = out_feat[(ch*resolution*resolution*resolution) + cell_id_flat];

					float dL_dsum = dL_dout_sums[cell_id_flat];
					float dL_dsdf = dL_dout_sdfs[cell_id_flat];
					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dfeat[ch] = dL_dout_feat[(ch*resolution*resolution*resolution) + cell_id_flat];

					float3 vec   = {cell.x-point.x, cell.y-point.y, cell.z-point.z};
					float  proj  = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;
					float  norm  = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
					float  theta = __expf(-1.0f * norm / (scale * scale));

					if (theta < 1e-4f)
						continue;

					float dtheta_i_dp_i_base = 2 * theta / (scale * scale);
					float dfeat_ch_dfeature_i_ch = theta / sum;

					// dL_dtheta_i
					float dL_dtheta_i = dL_dsdf * ((proj - sdf) / sum) + dL_dsum;
					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dtheta_i += dL_dfeat[ch] * (features[point_id*CHANNELS+ch] - feat[ch]) / sum;

					// dL_dproj_i
					float dL_dproj_i = dL_dsdf * (theta / sum);

					// dL_dfeature_i
					for (int ch = 0; ch < CHANNELS; ch++)
						dL_dfeature_i[ch] += dL_dfeat[ch] * dfeat_ch_dfeature_i_ch;

					// dL_dp_i
					dL_dp_i_x += dL_dtheta_i * (dtheta_i_dp_i_base * vec.x) + dL_dproj_i * (- normal.x);
					dL_dp_i_y += dL_dtheta_i * (dtheta_i_dp_i_base * vec.y) + dL_dproj_i * (- normal.y);
					dL_dp_i_z += dL_dtheta_i * (dtheta_i_dp_i_base * vec.z) + dL_dproj_i * (- normal.z);

					// dL_dn_i
					dL_dn_i_x += dL_dproj_i * vec.x;
					dL_dn_i_y += dL_dproj_i * vec.y;
					dL_dn_i_z += dL_dproj_i * vec.z;

					// dL_ds_i
					dL_ds_i   += dL_dtheta_i * (dtheta_i_dp_i_base * norm / scale);

				}
			}
		}
	}

	if (all_valid_flag != 0)
	{
		atomicAdd(&dL_dmeans3D[point_id*3 + 0], dL_dp_i_x);
		atomicAdd(&dL_dmeans3D[point_id*3 + 1], dL_dp_i_y);
		atomicAdd(&dL_dmeans3D[point_id*3 + 2], dL_dp_i_z);

		atomicAdd(&dL_dnormals[point_id*3 + 0], dL_dn_i_x);
		atomicAdd(&dL_dnormals[point_id*3 + 1], dL_dn_i_y);
		atomicAdd(&dL_dnormals[point_id*3 + 2], dL_dn_i_z);
		
		atomicAdd(&dL_dscales[point_id], dL_ds_i);

		for (int ch = 0; ch < CHANNELS; ch++)
			atomicAdd(&dL_dfeatures[point_id*CHANNELS + ch], dL_dfeature_i[ch]);
	}
}

void BACKWARD::splat(
	const dim3 grid, 
	const dim3 block,
	const int num_computed,
	const int num_phases,
	const int num_points,
	const int resolution, 
	const float* means3D,
	const float* normals,
	const float* scales, 
	const float* features,
	const uint2* phases_ranges,
	const uint32_t* phases_tiles,
	const uint32_t* point_list,
	const uint32_t* tiles_list,
	const int* grad_flags,
	const float* out_sums,
	const float* out_sdfs,
	const float* out_feat,
	const float* dL_dout_sums,
	const float* dL_dout_sdfs,
	const float* dL_dout_feat,
	float* dL_dmeans3D,
	float* dL_dnormals,
	float* dL_dscales,
	float* dL_dfeatures)
{
	splatCUDA<NUM_CHANNELS> << <(num_computed + 255) / 256, 256 >> >(
		num_computed,
		num_points,
		resolution, 
		means3D,
		normals,
		scales, 
		features,
		phases_ranges,
		phases_tiles,
		point_list,
		tiles_list,
		grad_flags,
		out_sums,
		out_sdfs,
		out_feat,
		dL_dout_sums,
		dL_dout_sdfs,
		dL_dout_feat,
		dL_dmeans3D,
		dL_dnormals,
		dL_dscales,
		dL_dfeatures);
}
