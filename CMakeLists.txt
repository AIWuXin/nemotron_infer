cmake_minimum_required(VERSION 3.21)
project(nemotron_infer LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")          # MSVC UTF-8 编码支持
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(WITH_GFLAGS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler=/utf-8")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)


if(MSVC)
    # 为 C++ 目标设置 /utf-8
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8>)

    # 为 CUDA 目标设置 -Xcompiler=/utf-8
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()


# ==========================================
# 自动下载 CPM.cmake
# ==========================================
set(
        THIRD_PARTY_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
set(
        CPM_DOWNLOAD_LOCATION
        "${THIRD_PARTY_DIR}/CPM.cmake"
)
if (NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    if (NOT EXISTS ${THIRD_PARTY_DIR})
        file(
                MAKE_DIRECTORY
                ${THIRD_PARTY_DIR}
        )
    endif ()
    file(
            DOWNLOAD
            https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            SHOW_PROGRESS
    )
endif ()
include(${CPM_DOWNLOAD_LOCATION})
# 让 CPM 依赖不创建顶层目标（不在 CLion 中显示）
set(CPM_USE_LOCAL_PACKAGES OFF)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)  # 允许选项被覆盖


CPMAddPackage(
        NAME GTest
        GITHUB_REPOSITORY google/googletest
        OPTIONS "BUILD_SHARED_LIBS OFF"
        VERSION 1.15.0
        EXCLUDE_FROM_ALL YES
)

CPMAddPackage(
        NAME glog
        GITHUB_REPOSITORY google/glog
        VERSION 0.7.1
        OPTIONS "BUILD_SHARED_LIBS OFF;BUILD_TESTING Off"
        EXCLUDE_FROM_ALL YES
)


# ============================================
# 1. 先找 Python（Development 组件提供 Python.h）
# ============================================
set(Python3_ROOT_DIR ${CMAKE_SOURCE_DIR}/.venv)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")


# ============================================
# 2. 找 pybind11
# ============================================
# 优先用 pip 安装的 pybind11 cmake 配置
execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR_AUTO
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
)

if(EXISTS "${pybind11_DIR_AUTO}")
    message(STATUS "Found pybind11 cmake dir: ${pybind11_DIR_AUTO}")
    set(pybind11_DIR ${pybind11_DIR_AUTO})
else()
    # 回退到手动指定
    set(pybind11_DIR ${Python3_SITELIB}/pybind11/share/cmake/pybind11)
endif()

find_package(pybind11 REQUIRED)
include(${CPM_DOWNLOAD_LOCATION})


# ============================================
# 3. CUDA
# ============================================
find_package(CUDAToolkit REQUIRED)

# 设置 CUDA 架构
# RTX 40系列: 89, RTX 50系列: 120 （原生FP8推理不支持SM86了）
set(CMAKE_CUDA_ARCHITECTURES 89 120)


# ============================================
# 4. CUDNN
# ============================================
option(USE_CUDNN "Enable cuDNN library" OFF)

if(USE_CUDNN)
    include(FindCUDNN)   # 注意：这里是 include，不是 find_package

    # 检查 cuDNN 是否被成功找到（脚本中如果没有设置 FOUND 变量，可以手动检查）
    if(NOT DEFINED cuDNN_INCLUDE_DIR OR NOT DEFINED cuDNN_LIBRARY_DIR)
        message(FATAL_ERROR "cuDNN not found. Please set CMAKE_INSTALL_PREFIX or CUDAToolkit_ROOT to the correct path.")
    endif()

    set(CUDNN_INCLUDE_DIRS ${cuDNN_INCLUDE_DIR})
    set(CUDNN_LIBRARIES cudnn)   # 链接主要的共享库目标（Windows 下为 cudnn）
endif()


# ============================================
# 5. 包含目录
# ============================================
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# 项目自身源码路径
include_directories(${CMAKE_SOURCE_DIR}/csrc)          # ops/
include_directories(${CMAKE_SOURCE_DIR}/csrc/ops)


# ============================================
# 6. 源文件和库
# ============================================
pybind11_add_module(
        nemotron_infer SHARED
        ${CMAKE_SOURCE_DIR}/nemotron_infer/csrc/include/binding.cu
)

target_link_libraries(nemotron_infer PRIVATE
        Python3::Python        # 关键：链接 Python 库
        pybind11::module       # 关键：链接 pybind11
        glog::glog
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
)

set_target_properties(
        nemotron_infer PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        OUTPUT_NAME "binding"
        VERSION 0.1.0
)

if (USE_CUDNN)
    target_link_libraries(nemotron_infer PRIVATE ${CUDNN_LIBRARIES})
    target_include_directories(nemotron_infer PRIVATE ${CUDNN_INCLUDE_DIRS})

    target_compile_definitions(nemotron_infer PRIVATE USE_CUDNN)
endif ()


# ============================================
# 7. 后处理 (仅cuda12的工具链, cuda13的目录应该略有不同)
# ============================================
set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/nemotron_infer/csrc")
file(GLOB CUDA_RUNTIME_DLL "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
add_custom_command(TARGET nemotron_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:nemotron_infer>" "${OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CUDA_RUNTIME_DLL}
        "${OUTPUT_DIR}"
        COMMENT "Copying binding.pyd and required CUDA DLLs to ${OUTPUT_DIR}"
        VERBATIM
)


# ==========================================
# 测试配置
# ==========================================
enable_testing()

# 收集测试文件
file(
        GLOB_RECURSE TEST_OPS CONFIGURE_DEPENDS
        tests/ops/*.cpp
        tests/ops/*.cu
)

add_executable(
        TEST_TENSOR
        ${DIR_SOURCES}
        tests/test_tensor.cu
)
target_include_directories(TEST_TENSOR PRIVATE
        ${CMAKE_SOURCE_DIR}/nemotron_infer/csrc/include          # tensor.h, allocator.cuh
)
target_link_libraries(
        TEST_TENSOR
        Python3::Python        # 关键：链接 Python 库
        pybind11::module       # 关键：链接 pybind11
        CUDA::cudart
        CUDA::cublas
        CUDA::cublasLt
        glog::glog
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
)
if (USE_CUDNN)
    target_link_libraries(TEST_TENSOR ${CUDNN_LIBRARIES})
    target_include_directories(TEST_TENSOR PRIVATE ${CUDNN_INCLUDE_DIRS})

    target_compile_definitions(TEST_TENSOR PRIVATE USE_CUDNN)
endif ()
add_custom_command(TARGET TEST_TENSOR POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:Python3::Python>        # 自动展开为 python312.dll 的完整路径
        $<TARGET_FILE_DIR:TEST_TENSOR>          # TEST_OPS.exe 所在目录
        COMMENT "Copying python312.dll next to TEST_OPS.exe"
)
set_target_properties(
        TEST_TENSOR PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
add_test(NAME TestTensor COMMAND TEST_TENSOR)
set_tests_properties(TestTensor PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

if (TEST_OPS)
    add_executable(
            TEST_OPS
            ${DIR_SOURCES}
            ${TEST_OPS}
    )
    target_include_directories(TEST_OPS PRIVATE
            ${CMAKE_SOURCE_DIR}/nemotron_infer/csrc/include          # tensor.h, allocator.cuh
    )
    target_link_libraries(
            TEST_OPS
            Python3::Python        # 关键：链接 Python 库
            pybind11::module       # 关键：链接 pybind11
            CUDA::cudart
            CUDA::cublas
            glog::glog
            GTest::gtest
            GTest::gtest_main
            GTest::gmock
    )
    if (USE_CUDNN)
        target_link_libraries(TEST_OPS ${CUDNN_LIBRARIES})
        target_include_directories(TEST_OPS PRIVATE ${CUDNN_INCLUDE_DIRS})

        target_compile_definitions(TEST_OPS PRIVATE USE_CUDNN)
    endif ()
    add_custom_command(TARGET TEST_OPS POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:Python3::Python>        # 自动展开为 python312.dll 的完整路径
            $<TARGET_FILE_DIR:TEST_OPS>          # TEST_OPS.exe 所在目录
            COMMENT "Copying python312.dll next to TEST_OPS.exe"
    )
    set_target_properties(
            TEST_OPS PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
    add_test(NAME TestOps COMMAND TEST_OPS)
    set_tests_properties(TestOps PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif ()
