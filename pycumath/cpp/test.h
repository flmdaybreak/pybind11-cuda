#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <sstream>
#include <iostream>

void map_array(pybind11::array_t<double> vec, double scalar);

// void map_array1(vector<double> vec, double scalar);

// void map_array_f(pybind11::array_t<float> vec, float scalar);