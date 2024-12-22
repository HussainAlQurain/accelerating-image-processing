# Implementation Details

This document provides an in-depth discussion of the pipeline:
- **Gaussian Blur**: Manually implemented 5x5 convolution filter.
- **Sobel Edge Detection**: Standard Sobel kernels along X and Y directions.
- **OpenACC Directives**: Explanation of `#pragma acc parallel loop collapse(2)` usage.
- **CUDA C/C++ Kernels**: Usage of device kernels and memory copies.
- **Numba Approach**: Pythonic GPU kernels, data transfers with `cuda.to_device`.

## Performance Tuning
- Data transfers minimization strategy
- Memory alignment considerations
- Nsight Systems profiling tips

## Additional References
- [OpenACC Official Site](https://www.openacc.org/)
- [Numba Documentation](https://numba.pydata.org/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
