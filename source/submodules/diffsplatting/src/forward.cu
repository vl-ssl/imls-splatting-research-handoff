#include <iostream>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "forward.h"
#include "auxiliary.h"


// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(
	int num_points, 
	int resolution,
	const dim3   grid,
	const float* means3D,
	const float* scales,
	const float* features,
	uint2* bbox,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= num_points)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;

	// Transform point by projecting
	float3 p_orig = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };

	// Compute extent in 3D space
	uint3 box_min, box_max;
	getBox3D(p_orig, scales[idx], box_min, box_max, resolution, grid);
	if ((box_max.x - box_min.x) + (box_max.y - box_min.y) + (box_max.z - box_min.z) == 0)
		return;

	// Store some useful helper data for the next steps.
	bbox[idx * 3 + 0] = {box_min.x, box_max.x}; // x
	bbox[idx * 3 + 1] = {box_min.y, box_max.y}; // y
	bbox[idx * 3 + 2] = {box_min.z, box_max.z}; // z
	tiles_touched[idx] = (box_max.x-box_min.x)*(box_max.y-box_min.y)*(box_max.z - box_min.z);
	
}


void FORWARD::preprocess(
	int num_points, 
	int resolution,
	const dim3 grid,
	const float* means3D,
	const float* scales,
	const float* features,
	uint2* bbox,
	uint32_t* tiles_touched)
{
	preprocessCUDA<NUM_CHANNELS> << <(num_points + 255) / 256, 256 >> > (
		num_points, 
		resolution,
		grid,
		means3D,
		scales,
		features,
		bbox,
		tiles_touched);
}


template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
splatCUDA(
	int num_phases,
	int resolution,
	const float* __restrict__ means3D,
	const float* __restrict__ normals,
	const float* __restrict__ scales,
	const float* __restrict__ features,
	const uint2* __restrict__ spa_ranges,
	const uint32_t* __restrict__ idx_tile,
	const uint32_t* __restrict__ point_list,
	int* __restrict__ out_counts,
	float* __restrict__ out_sums,
	float* __restrict__ out_sdfs,
	float* __restrict__ out_feat)

{	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();

	int phase_id  = block.group_index().x;
	int thread_id = block.thread_index().x;

	uint32_t tile_id = idx_tile[phase_id];
	uint2    range   = spa_ranges[phase_id];

	uint32_t num_blocks_x = (resolution + BLOCK_X - 1) / BLOCK_X;
	uint32_t num_blocks_y = (resolution + BLOCK_Y - 1) / BLOCK_Y;

	int block_idz = (int)( tile_id / (num_blocks_y*num_blocks_x));
	int block_idy = (int)((tile_id - block_idz*(num_blocks_y*num_blocks_x)) / num_blocks_x);
	int block_idx = (int)( tile_id - block_idz*(num_blocks_y*num_blocks_x) - block_idy*num_blocks_x);

	int thread_idz = (int)(thread_id / (BLOCK_Y*BLOCK_X));
	int thread_idy = (int)((thread_id - thread_idz*(BLOCK_Y*BLOCK_X)) / BLOCK_X);
	int thread_idx = (int)( thread_id - thread_idz*(BLOCK_Y*BLOCK_X) - thread_idy*BLOCK_X);	

	uint3 cell_id_min = {block_idx * BLOCK_X, block_idy * BLOCK_Y, block_idz * BLOCK_Z};
	uint3 cell_id_max = {min(cell_id_min.x+BLOCK_X, resolution), min(cell_id_min.y+BLOCK_Y, resolution), min(cell_id_min.z+BLOCK_Z, resolution) };

	uint3    cell_id      = { cell_id_min.x + thread_idx, cell_id_min.y + thread_idy, cell_id_min.z + thread_idz };
	uint32_t cell_id_flat = cell_id.z * resolution * resolution + cell_id.y * resolution + cell_id.x;

	float cell_x = (float)cell_id.x / (resolution-1);
	float cell_y = (float)cell_id.y / (resolution-1);
	float cell_z = (float)cell_id.z / (resolution-1);

	bool inside = cell_id.x < resolution && cell_id.y < resolution && cell_id.z < resolution;
	bool done = !inside;

	int toDo = range.y - range.x;

	float theta_sum = 0;
	float theta_proj_sum = 0;
	float theta_feat_sum[CHANNELS] = { 0 };

	__shared__ int    collected_id[LEN_PHASE];
	__shared__ float  collected_scales[LEN_PHASE];
	__shared__ float3 collected_means3D[LEN_PHASE];
	__shared__ float3 collected_normals[LEN_PHASE];

	if (range.x + thread_id < range.y)
	{
		int coll_id = point_list[range.x + thread_id];
		collected_id[thread_id]      = coll_id;
		collected_scales[thread_id]  = scales[coll_id];
		collected_means3D[thread_id] = {means3D[coll_id*3], means3D[coll_id*3+1], means3D[coll_id*3+2]};
		collected_normals[thread_id] = {normals[coll_id*3], normals[coll_id*3+1], normals[coll_id*3+2]};
	}
	block.sync();

	int valid_point_num = 0;
	for (int i = 0; i < toDo; i++)
	{	
		float3 point  = collected_means3D[i];
		float3 normal = collected_normals[i];
		float  scale  = collected_scales[i];

		float3 vec   = {cell_x-point.x, cell_y-point.y, cell_z-point.z};
		float  proj  = vec.x * normal.x + vec.y * normal.y + vec.z * normal.z;
		float  norm  = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
		float  theta = __expf(-1.0f * norm / (scale * scale));

		if (theta < 1e-4f)
			continue;
		
		valid_point_num += 1;

		theta_sum      += theta;
		theta_proj_sum += theta * proj;
		for (int ch = 0; ch < CHANNELS; ch++)
			theta_feat_sum[ch] += theta * features[collected_id[i]*CHANNELS+ch];
	}

	if ((theta_sum >= 1e-4f) && inside && (valid_point_num > 0))
	{	
		atomicAdd(&out_counts[cell_id_flat], valid_point_num);
		atomicAdd(&out_sums[cell_id_flat], theta_sum);
		atomicAdd(&out_sdfs[cell_id_flat], theta_proj_sum);
		for (int ch = 0; ch < CHANNELS; ch++)
			atomicAdd(&out_feat[(ch*resolution*resolution*resolution) + cell_id_flat], theta_feat_sum[ch]);
	}
}

void FORWARD::splat(
	const dim3 grid, 
	const dim3 block,
	int num_phases,
	int resolution, 
	const float* means3D,
	const float* normals,
	const float* scales, 
	const float* features,
	const uint2* spa_ranges,
	const uint32_t* idx_tile,
	const uint32_t* point_list,
	int* out_counts,
	float* out_sums,
	float* out_sdfs,
	float* out_feat)
{	
	splatCUDA<NUM_CHANNELS> << <num_phases, block.x*block.y*block.z >> > (
		num_phases,
		resolution,
		means3D,
		normals,
		scales,
		features,
		spa_ranges,
		idx_tile,
		point_list,
		out_counts,
		out_sums,
		out_sdfs,
		out_feat);
}


template <uint32_t CHANNELS>
__global__ void 
weightingCUDA(
	int resolution,
	const int* __restrict__ out_counts,
	float* __restrict__ out_sums,
	float* __restrict__ out_sdfs,
	float* __restrict__ out_feat)
{	
	auto thread_idx = cg::this_grid().thread_rank();

	int num_cell = resolution * resolution * resolution;
	if (thread_idx >= num_cell)
		return;

	float out_sum = out_sums[thread_idx];
	float out_sdf = out_sdfs[thread_idx];

	if ((out_counts[thread_idx] > 1) && (out_sum >= 1e-4f))
	{
		out_sdfs[thread_idx] = out_sdf / out_sum;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_feat[(ch*num_cell) + thread_idx] = out_feat[(ch*num_cell) + thread_idx] / out_sum;
	}
	else
	{
		out_sums[thread_idx] = 0.0f;
		out_sdfs[thread_idx] = 0.0f;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_feat[(ch*num_cell) + thread_idx] = 0.0f;
	}
}

void FORWARD::weighting(
	int resolution,
	const int* out_counts,
	float* out_sums,
	float* out_sdfs,
	float* out_feat)
{	
	weightingCUDA<NUM_CHANNELS> << <(resolution*resolution*resolution + 255)/256, 256 >> > (
		resolution,
		out_counts,
		out_sums,
		out_sdfs,
		out_feat);
}



