cmake_minimum_required(VERSION 3.5)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
  message(CMAKE_CUDA_ARCHITECTURES)
endif()

project(PicSFML_CUDA LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(EXECUTABLE_NAME "PicSFML_CUDA")
set(homeDir "/home/geotyper")

find_package(SFML 2.5.1 COMPONENTS graphics)
set (INCLUDES ${INCLUDES} ${SFML_INCLUDE_DIRS})
set (LIBS ${LIBS} ${SFML_LIBRARIES})

set (IMGUI_DIR "/home/geotyper/CPP_lib/imGui/imgui-master")
find_package(ImGui-SFML REQUIRED)


include_directories(${IMGUISFML_DIR})
include_directories(${IMGUI_DIR})
include_directories(${INCLUDE_SDIR})
link_directories(${INCLUDE_SDIR})


set(Head_files
    graphmodule2d.hpp
    checkCudaCall.hpp
    mainloop.hpp
    FpsCounter.hpp
   # transferViaPBO.hpp

    CudaAutoBuffer.hpp
    CudaEventTimer.hpp

    cudasolver.hpp
    jorge.hpp
    tinycolormap.hpp

    cudahelper.hpp
    HelperStructs.hpp

    )

set(Source_files
   # transferViaPBO.cpp
    graphmodule2d.cpp
    mainloop.cpp

    CudaAutoBuffer.cpp
    CudaEventTimer.cpp
    cudasolver.cu
    cudahelper.cu

    main.cpp
    )

set(ALL_FILES
    ${Head_files}
    ${Source_files}
    ${INCLUDES}

    )

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

#set(OpenGL_GL_PREFERENCE LEGACY)

#set(Eigen_include "/home/geotyper/CPP_lib/Eigen/Eigen")
#include_directories(${Eigen_include})

set(Cuda_common_include "/usr/local/cuda/samples/common/inc")
include_directories(${Cuda_common_include})

message(${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})


add_executable(${EXECUTABLE_NAME} ${ALL_FILES})

target_include_directories(${EXECUTABLE_NAME} PRIVATE ${INCLUDES} ${EIGEN3_INCLUDE_DIR} )

set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES})
#target_link_libraries (${EXECUTABLE_NAME} Eigen3::Eigen)

target_link_libraries(${EXECUTABLE_NAME} ${SDL2_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME} ${GLUT_LIBRARIES})

target_link_libraries(${EXECUTABLE_NAME} sfml-graphics sfml-audio)
target_link_libraries(${EXECUTABLE_NAME} ImGui-SFML::ImGui-SFML)
