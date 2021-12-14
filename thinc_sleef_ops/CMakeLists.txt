string(APPEND CMAKE_C_FLAGS " -fPIC")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/include)

add_subdirectory(../libcpuid ${CMAKE_BINARY_DIR}/libcpuid)
set_property(TARGET cpuid PROPERTY FOLDER "dependencies")
link_directories(${CMAKE_BINARY_DIR}/libcpuid/lib)

add_subdirectory(../sleef ${CMAKE_BINARY_DIR}/sleef)
set_property(TARGET sleef PROPERTY FOLDER "dependencies")
link_directories(${CMAKE_BINARY_DIR}/sleef/lib)

get_directory_property(COMPILER_SUPPORTS_AVX DIRECTORY ../sleef DEFINITION COMPILER_SUPPORTS_AVX)
get_directory_property(COMPILER_SUPPORTS_AVX512F DIRECTORY ../sleef DEFINITION COMPILER_SUPPORTS_AVX512F)
get_directory_property(COMPILER_SUPPORTS_NEON DIRECTORY ../sleef DEFINITION COMPILER_SUPPORTS_ADVSIMD)
get_directory_property(COMPILER_SUPPORTS_SSE2 DIRECTORY ../sleef DEFINITION COMPILER_SUPPORTS_SSE2)

add_cython_target(sleef_array CXX)

set(SIMD_ARRAY_SOURCES
  simd_array/array_scalar.cpp
  simd_array/dispatch.cpp)

if(COMPILER_SUPPORTS_NEON)
  list(APPEND SIMD_ARRAY_SOURCES simd_array/array_neon.cpp)
  add_compile_definitions(COMPILER_SUPPORTS_NEON)
endif()

if(COMPILER_SUPPORTS_AVX)
  list(APPEND SIMD_ARRAY_SOURCES simd_array/array_avx.cpp)
  set_source_files_properties(simd_array/array_avx.cpp PROPERTIES COMPILE_FLAGS -mavx)
  add_compile_definitions(COMPILER_SUPPORTS_AVX)
endif()

if(COMPILER_SUPPORTS_AVX512F)
  list(APPEND SIMD_ARRAY_SOURCES simd_array/array_avx512.cpp)
  set_source_files_properties(simd_array/array_avx512.cpp PROPERTIES COMPILE_FLAGS -mavx512f)
  add_compile_definitions(COMPILER_SUPPORTS_AVX512F)
endif()

if(COMPILER_SUPPORTS_SSE2)
  list(APPEND SIMD_ARRAY_SOURCES simd_array/array_sse.cpp)
  add_compile_definitions(COMPILER_SUPPORTS_SSE2)
endif()

add_library(sleef_array MODULE ${sleef_array} ${SIMD_ARRAY_SOURCES})
target_link_libraries(sleef_array sleef)
target_link_libraries(sleef_array cpuid)
python_extension_module(sleef_array)

add_cython_target(sleef_ops CXX)
add_library(sleef_ops MODULE ${sleef_ops})
add_dependencies(sleef_ops sleef_array)
python_extension_module(sleef_ops)

install(TARGETS sleef_array LIBRARY DESTINATION thinc_sleef_ops)
install(TARGETS sleef_ops LIBRARY DESTINATION thinc_sleef_ops)
