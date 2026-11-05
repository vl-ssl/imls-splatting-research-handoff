#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SplatterForward(
	const torch::Tensor& means3D,
    const torch::Tensor& normals,
    const torch::Tensor& scales,
    const torch::Tensor& features,
    const int resolution);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
SplatterBackward(
    const int num_computed,
    const int num_phases,
    const int resolution,
	const torch::Tensor& means3D,
    const torch::Tensor& normals,
    const torch::Tensor& scales,
    const torch::Tensor& features,
    const torch::Tensor& geoBuffer,
    const torch::Tensor& binBuffer,
    const torch::Tensor& voxBuffer,
    const torch::Tensor& spaBuffer,
    const torch::Tensor& out_sums,
    const torch::Tensor& out_sdfs,
    const torch::Tensor& out_feat,
    const torch::Tensor& dL_dout_sums,
    const torch::Tensor& dL_dout_sdfs,
    const torch::Tensor& dL_dout_feat
	);


