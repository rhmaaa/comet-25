# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)


# project(outliers_cuda CUDA CXX)

# # set(PYTORCH_ROOT "/root/miniconda3/envs/mixq/lib/python3.10/site-packages/torch")
# # list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_ROOT}/share/cmake/Torch/")
# # find_package(Torch REQUIRED CONFIG)
# # find_package(Torch REQUIRED)
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/torch")

# #This is needed to be able to find the mkl and other dependent libraries
# link_directories("$ENV{CONDA_PREFIX}/lib")
# set(ENV{MKLROOT} "$ENV{CONDA_PREFIX}/lib")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# # find_package(Torch REQUIRED)
# find_package(CUDA REQUIRED)
# find_package(Python REQUIRED COMPONENTS Development)
# # Set CUDA flags
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")  # Adjust compute capability as needed

# # Add the executable
# add_executable(outliers_cuda 
#     comet.cu
# )
# #We need to add pthread and omp manually here
# target_link_libraries(outliers_cuda "${TORCH_LIBRARIES}" pthread omp)

# set_property(TARGET outliers_cuda PROPERTY CXX_STANDARD 14)


if (CUDA_FOUND)

endif ()

set(_C_LIBRARIES "")

add_subdirectory(w4amix)
# add_subdirectory(symmetric)
# add_subdirectory(asymmetric)

pybind11_add_module(_C MODULE THIN_LTO binding.cpp)

target_link_libraries(
        _C PUBLIC
        ${TORCH_LIBRARIES}
        ${_C_LIBRARIES}
)
