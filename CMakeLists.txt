cmake_minimum_required(VERSION 3.1)
project(project LANGUAGES CXX)
find_package(OpenCV REQUIRED)

set (opencv_manual "/usr/include")
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
        STRING "Choose the type of build." FORCE)
endif()

add_executable(upscale_image app/test_conv2d.cpp lib/espcn.cpp )
set_property(TARGET upscale_image PROPERTY CXX_STANDARD 17)


target_link_libraries(upscale_image ${OpenCV_LIBS})

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${opencv_manual}
    )


message(${OpenCV_INCLUDE_DIRS})
install(TARGETS upscale_image DESTINATION bin)
install(PROGRAMS demo DESTINATION bin)
install(FILES weights.csv DESTINATION files)
install(FILES tests_images/coke.jpg DESTINATION files)