#ifndef CUDA_SPLATTER_FORWARD_H_INCLUDED
#define CUDA_SPLATTER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{

	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(
		int num_points, 
		int resolution,
		const dim3 grid,
		const float* means3D,
		const float* scales,
		const float* features,
		uint2* bbox,
		uint32_t* tiles_touched);

	void splat(
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
		float* out_feat);

	void weighting(
		int resolution,
		const int* out_counts,
		float* out_sums,
		float* out_sdfs,
		float* out_feat);
}

#endif
