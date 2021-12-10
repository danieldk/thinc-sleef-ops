string(APPEND CMAKE_C_FLAGS " -fPIC")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/include)

add_subdirectory(../libcpuid ${CMAKE_BINARY_DIR}/libcpuid)
set_property(TARGET cpuid PROPERTY FOLDER "dependencies")
link_directories(${CMAKE_BINARY_DIR}/libcpuid/lib)

add_cython_target(cpu_id CXX)
add_library(cpu_id MODULE ${cpu_id})
target_link_libraries(cpu_id cpuid)
python_extension_module(cpu_id)

add_subdirectory(../sleef ${CMAKE_BINARY_DIR}/sleef)
set_property(TARGET sleef PROPERTY FOLDER "dependencies")
link_directories(${CMAKE_BINARY_DIR}/sleef/lib)

set_source_files_properties(array_avx.cpp PROPERTIES COMPILE_FLAGS -mavx)
set_source_files_properties(array_avx512.cpp PROPERTIES COMPILE_FLAGS -mavx512f)

add_cython_target(sleef_ops CXX)
add_library(sleef_ops MODULE ${sleef_ops} array_scalar.cpp array_avx.cpp array_avx512.cpp array_sse.cpp)
target_link_libraries(sleef_ops sleef)
python_extension_module(sleef_ops)

install(TARGETS cpu_id LIBRARY DESTINATION thinc_sleef_ops)
install(TARGETS sleef_ops LIBRARY DESTINATION thinc_sleef_ops)
