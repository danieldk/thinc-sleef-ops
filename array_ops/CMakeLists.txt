
string(APPEND CMAKE_C_FLAGS " -fPIC")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build sleef static" FORCE)
set(BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)

add_subdirectory(../sleef ${CMAKE_BINARY_DIR}/sleef)
set_property(TARGET sleef PROPERTY FOLDER "dependencies")
link_directories(${CMAKE_BINARY_DIR}/sleef/lib)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/include)

SET_SOURCE_FILES_PROPERTIES(array_avx.cpp PROPERTIES COMPILE_FLAGS -mavx)
SET_SOURCE_FILES_PROPERTIES(array_avx512.cpp PROPERTIES COMPILE_FLAGS -mavx512f)

add_cython_target(array_ops CXX)
add_library(array_ops MODULE ${array_ops} array_scalar.cpp array_avx.cpp array_avx512.cpp array_sse.cpp)
target_link_libraries(array_ops sleef)
python_extension_module(array_ops)

install(TARGETS array_ops LIBRARY DESTINATION thinc_sleef_ops)
