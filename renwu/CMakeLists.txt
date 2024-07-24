cmake_minimum_required(VERSION 3.1)

project(renwu1)

find_package(OpenCV REQUIRED)

add_executable(renwu1 main.cpp)
target_link_libraries(renwu1 PRIVATE ${OpenCV_LIBS})
