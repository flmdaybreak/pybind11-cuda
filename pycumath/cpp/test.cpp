#include "test.h"
#include "../cuda/gpu_library.cuh"

void map_array(pybind11::array_t<double> vec, double scalar)
{
  pybind11::buffer_info ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  int size = ha.shape[0];
  double* ptr = reinterpret_cast<double*>(ha.ptr);
  gpu_task1(ptr, size, scalar);
}



/*
void map_array1(vector<double> vec, double scalar)
{
  // pybind11::buffer_info ha = vec.request();

  // if (ha.ndim != 1) {
  //   std::stringstream strstr;
  //   strstr << "ha.ndim != 1" << std::endl;
  //   strstr << "ha.ndim: " << ha.ndim << std::endl;
  //   throw std::runtime_error(strstr.str());
  // }

  int size = vec.size();
  int size_bytes = size*sizeof(double);
  double *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  double* ptr = reinterpret_cast<double*>(vec.data());
  error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel(gpu_ptr, scalar, size);

  error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

void map_array_f(pybind11::array_t<float> vec, float scalar)
{
  pybind11::buffer_info ha = vec.request();

  if (ha.ndim != 1) {
    std::stringstream strstr;
    strstr << "ha.ndim != 1" << std::endl;
    strstr << "ha.ndim: " << ha.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  int size = ha.shape[0];
  int size_bytes = size*sizeof(float);
  float *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  float* ptr = reinterpret_cast<float*>(ha.ptr);
  error = cudaMemcpy(gpu_ptr, ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel_f(gpu_ptr, scalar, size);

  error = cudaMemcpy(ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}
*/
// PYBIND11_MODULE(gpu_library, m)
// {
//   m.def("multiply_with_scalar_f", map_array_f);
//   m.def("multiply_with_scalar1", map_array1);
//   m.def("multiply_with_scalar", map_array);
// }
