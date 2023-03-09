#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

using namespace std;


__global__ void kernel
(double *vec, double scalar, int num_elements);

void run_kernel
(double *vec, double scalar, int num_elements);

__global__ void kernel_f
(float *vec, float scalar, int num_elements);

void run_kernel_f(float *vec, float scalar, int num_elements);

void gpu_task1(double *vec_ptr, int size, double scalar);

void gpu_task2(float *vec_ptr, int size, double scalar);


// void map_array(pybind11::array_t<double> vec, double scalar);

// void map_array1(vector<double> vec, double scalar);

// void map_array_f(pybind11::array_t<float> vec, float scalar);
