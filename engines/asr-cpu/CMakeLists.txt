# Kitten ASR CPU engine: libkasr (include/kasr.h) on the vendored llama.cpp libraries (ggml, libllama, libmtmd).
#
# Installs into kitten_inference/_asr_cpu/lib (CMake install component "python", the one the wheel takes):
#   Linux / Windows: kasr + llama + mtmd + ggml + ggml-base, and one ggml-cpu-<variant> library per CPU generation
#                    (x86-64: AVX2 and newer; arm64: NEON). ggml loads the best one
#                    for the CPU it runs on, so one wheel serves every CPU of the platform.
#   macOS:           the same libraries with the CPU and Metal backends built in (shaders embedded).
# Standalone: cmake -S engines/asr-cpu -B build && cmake --build build -j && cmake --install build --component python --prefix out
cmake_minimum_required(VERSION 3.18)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(kitten_asr_cpu C CXX)
endif()
if(WIN32)
    set(_kasr_arch "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}")
    if(NOT _kasr_arch)
        set(_kasr_arch "${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    string(TOLOWER "${_kasr_arch}" _kasr_arch)
    if(NOT _kasr_arch MATCHES "^(x64|amd64|x86_64)$")
        message(FATAL_ERROR "The speech recognition CPU engine on Windows requires an x64 build.")
    endif()
endif()

set(KASR_INSTALL_DIR "kitten_inference/_asr_cpu/lib" CACHE STRING "install folder of the ASR CPU libraries")

# llama.cpp as a library only: no tools, server, tests, network or video
set(BUILD_SHARED_LIBS ON)
set(LLAMA_STANDALONE OFF)
set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_BUILD_UI OFF)
set(LLAMA_BUILD_MTMD ON)
set(LLAMA_OPENSSL OFF)
set(LLAMA_SUBPROCESS OFF)
set(LLAMA_ALL_WARNINGS OFF)
set(MTMD_VIDEO OFF)
set(GGML_NATIVE OFF)          # never tune for the build machine: the wheel runs elsewhere
set(GGML_OPENMP OFF)          # ggml's own thread pool: no OpenMP runtime to ship
set(GGML_BLAS OFF)
set(GGML_CUDA OFF)
set(GGML_VULKAN OFF)
set(GGML_LLAMAFILE ON)        # tiled f16/bf16/q8_0 matrix kernels on the CPU
if(APPLE)
    set(GGML_METAL ON)
    set(GGML_METAL_EMBED_LIBRARY ON)
    set(GGML_BACKEND_DL OFF)
    set(GGML_ACCELERATE OFF)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else()
    set(GGML_METAL OFF)
    set(GGML_BACKEND_DL ON)
    option(KASR_CPU_ALL_VARIANTS "one CPU backend per CPU generation (off: one backend for the build machine's CPU, for development)" ON)
    set(GGML_CPU_ALL_VARIANTS ${KASR_CPU_ALL_VARIANTS})
    set(GGML_CPU_X86_AVX2_MIN ON)    # x86: AVX2 and newer only, as the rest of the package
    set(GGML_CPU_ARM_NEON_ONLY ON)   # arm64 Linux: NEON variants only (see ggml/src/CMakeLists.txt, "kitten:")
    if(NOT KASR_CPU_ALL_VARIANTS)
        set(GGML_NATIVE ON)
    endif()
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(llama.cpp)

add_library(kasr SHARED src/kasr.cpp)
target_include_directories(kasr PUBLIC include)
target_link_libraries(kasr PRIVATE llama mtmd ggml)
target_compile_features(kasr PRIVATE cxx_std_17)
set_target_properties(kasr PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(kasr PRIVATE dl)
endif()

# One file per library in the wheel: no versioned names or symlinks (a wheel would store each symlink as a copy).
set(KASR_LIBS kasr llama mtmd ggml ggml-base)
foreach(t IN LISTS KASR_LIBS)
    set_property(TARGET ${t} PROPERTY VERSION)
    set_property(TARGET ${t} PROPERTY SOVERSION)
    set_property(TARGET ${t} PROPERTY NO_SONAME OFF)
endforeach()
# The backends too (ggml-cpu, ggml-metal on macOS): with a version, macOS installs libggml-metal.1.0.0.dylib plus
# libggml-metal.1.dylib and the wheel holds both (+1.4 MB compressed, macOS arm64 wheel built in a macOS virtual machine).
foreach(t IN LISTS GGML_AVAILABLE_BACKENDS)
    set_property(TARGET ${t} PROPERTY VERSION)
    set_property(TARGET ${t} PROPERTY SOVERSION)
endforeach()
if(APPLE)
    list(APPEND KASR_LIBS ${GGML_AVAILABLE_BACKENDS})
endif()

install(TARGETS ${KASR_LIBS}
        LIBRARY DESTINATION ${KASR_INSTALL_DIR} COMPONENT python NAMELINK_SKIP
        RUNTIME DESTINATION ${KASR_INSTALL_DIR} COMPONENT python
        ARCHIVE DESTINATION ${KASR_INSTALL_DIR}/../_unused COMPONENT unused
        PUBLIC_HEADER DESTINATION ${KASR_INSTALL_DIR}/../_unused COMPONENT unused
        PRIVATE_HEADER DESTINATION ${KASR_INSTALL_DIR}/../_unused COMPONENT unused)
if(NOT APPLE)
    install(TARGETS ${GGML_AVAILABLE_BACKENDS}
            LIBRARY DESTINATION ${KASR_INSTALL_DIR} COMPONENT python
            RUNTIME DESTINATION ${KASR_INSTALL_DIR} COMPONENT python)
endif()
install(FILES llama.cpp/LICENSE DESTINATION ${KASR_INSTALL_DIR}/.. RENAME LICENSE-llama.cpp COMPONENT python)
