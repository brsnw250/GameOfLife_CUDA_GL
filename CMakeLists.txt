cmake_minimum_required(VERSION 3.22)
project(gol_cuda_gl LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(${PROJECT_NAME} main.cpp cuda_ops.cuh cuda_ops.cu)

#CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#CUDA Toolkit
find_package(CUDAToolkit REQUIRED)


#OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES})

#GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(lib/glfw)
target_link_libraries(${PROJECT_NAME} glfw)

#GLAD
add_subdirectory(lib/glad/cmake)
glad_add_library(glad_gl_comp_33 REPRODUCIBLE API gl:compatibility=3.3)
target_link_libraries(${PROJECT_NAME} glad_gl_comp_33)


if(MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup")
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif()


if (${CMAKE_BUILD_TYPE} MATCHES "Release")
    set(${CMAKE_CXX_FLAGS} "${CMAKE_CXX_FLAGS} /O2")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            --relocatable-device-code=true>)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50;52;61;75;86")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

