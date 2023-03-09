#include "gpu_library.cuh"

__global__ void kernel
(double *vec, double scalar, int num_elements)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}

void run_kernel
(double *vec, double scalar, int num_elements)
{
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
  
  kernel<<<dimGrid, dimBlock>>>
    (vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}


__global__ void kernel_f
(float *vec, float scalar, int num_elements)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    vec[idx] = vec[idx] * scalar;
  }
}

void run_kernel_f
(float *vec, float scalar, int num_elements)
{
  dim3 dimBlock(256, 1, 1);
  dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
  
  kernel_f<<<dimGrid, dimBlock>>>
    (vec, scalar, num_elements);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}

void gpu_task1(double *vec_ptr, int size, double scalar)
{

  int size_bytes = size*sizeof(double);
  double *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaMemcpy(gpu_ptr, vec_ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel(gpu_ptr, scalar, size);

  error = cudaMemcpy(vec_ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}


void gpu_task2(float *vec_ptr, int size, double scalar)
{

  int size_bytes = size*sizeof(double);
  float *gpu_ptr;
  cudaError_t error = cudaMalloc(&gpu_ptr, size_bytes);

  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaMemcpy(gpu_ptr, vec_ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_kernel_f(gpu_ptr, scalar, size);

  error = cudaMemcpy(vec_ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(gpu_ptr);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
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
