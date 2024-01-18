# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project (shoal-of-fish)

find_package(OpenGL REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 52 61 75)
find_package(Sanitizers)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 17)

#file(GLOB_RECURSE IMGUI_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/lib/imgui-cmake/imgui/*.cpp")

set(IMGUI_SOURCES
    lib/imgui/imgui.cpp
    lib/imgui/imgui_demo.cpp
    lib/imgui/imgui_draw.cpp
    lib/imgui/imgui_widgets.cpp
    lib/imgui/imgui_tables.cpp
    lib/imgui/backends/imgui_impl_glfw.cpp
    lib/imgui/backends/imgui_impl_opengl3.cpp
    ${GPU_SOURCES}
)

file(GLOB_RECURSE CPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
file(GLOB_RECURSE GPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu")

add_executable(shoal-of-fish ${CPU_SOURCES} ${IMGUI_SOURCES})
target_compile_options(shoal-of-fish PRIVATE -Wall)

file(COPY src/shaders DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL)
add_subdirectory(lib/glm EXCLUDE_FROM_ALL)

target_include_directories(shoal-of-fish PRIVATE lib/imgui)
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

set_property(TARGET shoal-of-fish PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(shoal-of-fish
  PRIVATE glfw
  PRIVATE libglew_static
  PRIVATE glm
)

target_include_directories(shoal-of-fish
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src
)


