cmake_minimum_required(VERSION 3.1)

project(shipin)
find_package(OpenCV REQUIRED)

add_executable(opencv_shipin main.cpp)
target_link_libraries(opencv_shipin PRIVATE ${OpenCV_LIBS})
