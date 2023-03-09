# use cmake version >= 3.12 so that to benefit from
# the new FindPython3 module
cmake_minimum_required(VERSION 3.12)

message("Using CMake version ${CMAKE_VERSION}")
if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    message("Using cmake policy CMP0104 OLD")
    cmake_policy(SET CMP0104 OLD)
else()
    message("Using cmake policy CMP0104 NEW")
    cmake_policy(SET CMP0104 NEW)
endif()

project(pybind11-cuda-test LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# Prevent from build in source tree
include(preventBuildInSource)

# Init build type: Release, Debug, ...
include(initBuildType)

# useful for debugging cmake
include(print_target_properties)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------#
# Cuda and Python configuration
#-------------------------------------------------------------------#

enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# make sure to find python from conda, if a conda virtual env is activated
set(Python3_FIND_VIRTUALENV "STANDARD")
find_package(Python3 COMPONENTS Interpreter Development)
#find_package(PythonInterp 3.6 REQUIRED)
#find_package(PythonLibs 3.6 REQUIRED)

# we also need extra stuff to make sure compile flags are correctly
# passed to nvcc / host compiler
include(protect_nvcc_flags)
include(protect_pthread_flag)

#-------------------------------------------------------------------#
# build some Cuda library
#-------------------------------------------------------------------#
find_package(Python REQUIRED COMPONENTS Development)
include_directories(${Python_INCLUDE_DIRS})
# add_subdirectory(src)
set(PYPSI_BASEDIR pycumath)
set(CUDA_SOURCES
    ${PYPSI_BASEDIR}/cuda/gpu_library.cu
    )
set(CPP_SOURCES
    ${PYPSI_BASEDIR}/cpp/test.cpp
    )

add_library(gpu_library SHARED
  ${CUDA_SOURCES})

# target_include_directories(gpu_library PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(gpu_library PUBLIC
  #${PYTHON3_PYTHON}
  cudart)



execute_process(COMMAND python3 -m pybind11 --cmakedir OUTPUT_VARIABLE PYBIND11_CMAKE_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 PATHS ${PYBIND11_CMAKE_PATH} REQUIRED)

add_library(pycumath ${CPP_SOURCES})
pybind11_add_module(_pycumath ${CPP_SOURCES} ${PYPSI_BASEDIR}/binding.cpp)

target_link_libraries(pycumath PUBLIC
  #${PYTHON3_PYTHON}
  # Python3::Python
  gpu_library
  # gpu_library
  )

target_link_libraries(_pycumath PUBLIC
  #${PYTHON3_PYTHON}
  # Python3::Python
  gpu_library
  # gpu_library
  )



# target_compile_options(_gpu_library PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_60>")

# add_library(pypsi  ${SOURCES})
# pybind11_add_module(_pypsi_cpp ${SOURCES} ${PYPSI_BASEDIR}/binding.cpp)

# set what comes before the library name, here no "lib"
set_target_properties(gpu_library PROPERTIES PREFIX "")
# set_target_properties(_gpu_library PROPERTIES PREFIX "")

#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} "
  "${CMAKE_CXX_COMPILER_VERSION} "
  "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("  CUDA Compiler      : ${CMAKE_CUDA_COMPILER}")
message("  CUDA Compiler exec : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA Compile flags : ${CMAKE_CUDA_FLAGS}")
message("  CUDA toolkit inc   : ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message("")
message("  Python3_found       : ${Python3_FOUND}")
message("  Python3 version     : ${Python3_VERSION}")
message("  Python3 interpreter : ${Python3_EXECUTABLE}")
message("  Python3 interp. Id  : ${Python3_INTERPRETER_ID}")
message("  Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message("  Python3_LIBRARY_DIRS: ${Python3_LIBRARY_DIRS}")
