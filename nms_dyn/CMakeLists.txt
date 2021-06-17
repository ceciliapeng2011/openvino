cmake_minimum_required(VERSION 2.8)

set(CMAKE_VERBOSE_MAKEFILE on)

set(TARGET_NAME "ngraph_nms_test")

PROJECT (nms_test)

find_package(ngraph REQUIRED )
find_package(InferenceEngine REQUIRED)
message(STATUS "ngraph variables : ${NGRAPH_LIBRARIES}, ${NGRAPH_INCLUDES}")
message(STATUS "IE variables : ${InferenceEngine_LIBRARIES}, ${InferenceEngine_INCLUDE_DIRS}")


set( SRC
    src/main.cpp    
)

set( HEADERS       
     #./../ngraph/core/reference/include/
     ${CMAKE_CURRENT_SOURCE_DIR}/include
     ${InferenceEngine_INCLUDE_DIRS}
)
INCLUDE_DIRECTORIES( ${HEADERS} )

#link_directories( ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_executable( ${TARGET_NAME} ${SRC} )
target_link_libraries(${TARGET_NAME} PRIVATE ${NGRAPH_LIBRARIES} ${InferenceEngine_LIBRARIES})
