# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
project (opengl-cmake)

find_package(OpenGL REQUIRED)

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
)
# Define MY_SOURCES to be a list of all the source files for my project
file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")
add_executable(opengl-cmake ${MY_SOURCES} ${IMGUI_SOURCES})
target_compile_options(opengl-cmake PRIVATE -Wall)

# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/vertex_core.glsl" ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/fragment_core.glsl" ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/box/fragment_box.glsl" "${CMAKE_CURRENT_BINARY_DIR}/box/fragment_box.glsl" COPYONLY)
# configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/box/vertex_box.glsl" "${CMAKE_CURRENT_BINARY_DIR}/box/vertex_box.glsl" COPYONLY)
file(COPY src/shaders DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


add_definitions(-DGLEW_STATIC)
add_subdirectory(lib/glfw EXCLUDE_FROM_ALL)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL)
add_subdirectory(lib/glm EXCLUDE_FROM_ALL)
add_subdirectory(lib/SOIL2 EXCLUDE_FROM_ALL)

target_include_directories(opengl-cmake PRIVATE lib/imgui)

target_link_libraries(opengl-cmake
  PRIVATE glfw
  PRIVATE libglew_static
  PRIVATE glm
  PRIVATE soil2
)

target_include_directories(opengl-cmake
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src
)
