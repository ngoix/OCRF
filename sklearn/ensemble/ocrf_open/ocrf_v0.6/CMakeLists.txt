cmake_minimum_required(VERSION 2.6)

project(ocrf)

#find_package(Boost 1.46 COMPONENTS filesystem unit_test_framework REQUIRED)
#message(status "** Boost Include: ${Boost_INCLUDE_DIR}")
#message(status "** Boost Libraries: ${Boost_LIBRARY_DIRS}")
#message(status "** Boost Libraries: ${Boost_LIBRARIES}")

#SET(Boost_ADDITIONAL_VERSIONS "1.40.0" "1.46" )
#set(Boost_ADDITIONAL_VERSIONS "1.46" "1.47" "1.48" "1.54")

#set(BOOST_ROOT "libs/boost_1_54_0")

#link_directories ( ${Boost_LIBRARY_DIRS} )
#include_directories ( ${Boost_INCLUDE_DIRS} )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BUILD_TYPE})
#set(OPENCV_LIB_PATH "/usr/local/include/opencv")

#FIND_PACKAGE(OpenCV)
#introuvable

FILE(GLOB source_files ${CMAKE_SOURCE_DIR} src/*.cpp)
FILE(GLOB header_files ${CMAKE_SOURCE_DIR} include/*.h)
SOURCE_GROUP("Source Files" FILES ${folder_source})
SOURCE_GROUP("Header Files" FILES ${folder_header})

SET (EXECUTABLE_NAME
ocrf
)

#SET(BOOST_INCLUDE
#libs/boost_1_54_0
#)

#SET(BOOST_LINKER
#libs/boost_1_54_0/stage/lib
#)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR})
#INCLUDE_DIRECTORIES(${BOOST_INCLUDE})
#INCLUDE_DIRECTORIES(${OPENCV_LIB_PATH})
#INCLUDE_DIRECTORIES(include)
#INCLUDE_DIRECTORIES("/usr/include/")

#INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
# include_directories(include)

#TARGET_LINK_LIBRARIES(forest cxcore cv highgui cvaux m)
#TARGET_LINK_LIBRARIES(classification ${OPENCV_LIBRARIES})

#LINK_DIRECTORIES(${BOOST_LINKER})


ADD_EXECUTABLE(${EXECUTABLE_NAME} ${source_files} ${header_files})

target_link_libraries (
  ${EXECUTABLE_NAME}
  ${Boost_LIBRARIES}
)
