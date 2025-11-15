#pragma once
#include "image/View.cuh"

__global__ void col_prefix_sum(device_view::View *image);
__global__ void row_prefix_sum(device_view::View *image, int N);