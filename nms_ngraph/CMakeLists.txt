cmake_minimum_required(VERSION 2.8)

set(CMAKE_VERBOSE_MAKEFILE on)

set(TARGET_NAME "ngraph_nms_test")

PROJECT (nms_test)
find_package(ngraph REQUIRED )
set( SRC
    src/multiclass_nms.cpp
    src/main.cpp    
)

set( HEADERS       
     include/multiclass_nms.hpp
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include )
link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
add_executable( ${TARGET_NAME} ${SRC} ${HEADERS} )

message(STATUS "ngraph variables : ${NGRAPH_LIBRARIES}, ${NGRAPH_INCLUDES}")
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES})
