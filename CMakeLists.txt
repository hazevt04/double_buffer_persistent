cmake_minimum_required( VERSION 2.8.0 )

# Unless your CMake version is at least 3.17,
# the 'Modern CMake' practices don't work well with 
# CUDA with respect to detecting GPU architecure
# and finding CUDA's include directory!
# This is done the 'old way'

# If I had a newer cmake, I could do:
# project( persistent_kernel LANGUAGES CXX CUDA )
# without losing half a day fighting cmake
project( double_buffer_persistent )

# Force nvcc AND g++/gcc to use C++11
# and make sure it doesn't use GNU std
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(CUDA_SEPARABLE_COMPILATION ON)

# Not needed if cmake version is greater than 3.17
find_package(CUDA QUIET REQUIRED)

# Selects GPU arch flags for nvcc based on target_CUDA_architectures
# Xavier is compute capability 7.2
# Support included for Pascal and Volta microarchitectures
#cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.0 6.1 6.2 7.0 7.2)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(INC_DIRS
   "include"
   ${CUDA_INCLUDE_DIRS}
   )

set(SRCS
   "src/main.cu"
   )

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

include_directories( "${INC_DIRS}" )

cuda_add_executable( ${EXEC_NAME} "${SRCS}" )
target_link_libraries( ${EXEC_NAME} ${CUDA_cudadevrt_LIBRARY} m )

