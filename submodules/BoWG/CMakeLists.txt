# 文件作用：配置 BoWG 编译目标、依赖和安装导出；隶属于构建配置模块。
cmake_minimum_required(VERSION 3.0)
project(BoWG)

if(POLICY CMP0057)
  cmake_policy(SET CMP0057 NEW)
endif()

include(CTest)

include(ExternalProject)
option(BUILD_BOWG "Build BoWG" ON)  
option(BUILD_DETECTOR "Build detector application" ON)
option(BUILD_PYTHON_BINDINGS "Build the pybind11 BoWG module" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Header files
set(BOWG_HDRS
  include/BoWG/BoWGDatabase.h
  include/BoWG/BoWGScoring.h
  include/BoWG/BoWGVector.h
  include/BoWG/GeometicUtils.h
  include/BoWG/BowVector.h
  include/BoWG/DBoW2.h
  include/BoWG/FBrief.h
  include/BoWG/FClass.h
  include/BoWG/FeatureVector.h
  include/BoWG/QueryResults.h
  include/BoWG/ScoringObject.h
  include/BoWG/TemplatedDatabase.h
  include/BoWG/TemplatedVocabulary.h
  include/BoWG/Parameters.h
  include/BoWG/BoWGDetector.h)

set(DUTILS_HDRS
  include/DUtils/Random.h
  include/DUtils/Timestamp.h
  include/DUtils/DUtils.h
  include/DUtils/DException.h)

set(DUTILSCV_HDRS
  include/DUtilsCV/Drawing.h
  include/DUtilsCV/Transformations.h
  include/DUtilsCV/Types.h
  include/DUtilsCV/DUtilsCV.h
  include/DUtilsCV/Geometry.h
  include/DUtilsCV/GUI.h
  include/DUtilsCV/IO.h
  include/DUtilsCV/Mat.h)

set(DVISION_HDRS
  include/DVision/BRIEF.h
  include/DVision/DVision.h
  include/DVision/FSolver.h
  include/DVision/BriefExtractor.h)

# Source files
set(SRCS 
  # BoWG sources
  src/BoWG/BoWGDatabase.cpp
  src/BoWG/BoWGScoring.cpp
  src/BoWG/BoWGVector.cpp
  src/BoWG/BowVector.cpp
  src/BoWG/FeatureVector.cpp
  src/BoWG/QueryResults.cpp
  src/BoWG/ScoringObject.cpp
  src/BoWG/FBrief.cpp
  src/BoWG/Parameters.cpp
  src/BoWG/BoWGDetector.cpp
  
  # DUtils sources
  src/DUtils/Random.cpp
  src/DUtils/Timestamp.cpp
  
  # DVision sources
  src/DVision/BRIEF.cpp
  src/DVision/FSolver.cpp
  src/DVision/BriefExtractor.cpp
  
  # VocabularyBinary
  src/VocabularyBinary.cpp)

set(DEPENDENCY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
set(DEPENDENCY_INSTALL_DIR ${DEPENDENCY_DIR}/install)

# Dependencies
find_package(OpenCV REQUIRED)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include/
  include/BoWG/
  include/DUtils/
  include/DVision/
)

if(BUILD_BOWG)
  set(LIB_SHARED "SHARED")
  if(WIN32)
    set(LIB_SHARED "STATIC")
  endif(WIN32)
  
  add_library(${PROJECT_NAME} ${LIB_SHARED} ${SRCS})
  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)

  add_executable(train_voc src/Train_Voc/train_voc.cpp)
  target_link_libraries(train_voc 
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
  )
  set_target_properties(train_voc PROPERTIES CXX_STANDARD 11)
endif(BUILD_BOWG)

if(BUILD_PYTHON_BINDINGS AND BUILD_BOWG)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
    OUTPUT_VARIABLE PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT PYBIND11_CMAKEDIR_RESULT EQUAL 0)
    message(FATAL_ERROR
      "pybind11 is not installed for ${Python3_EXECUTABLE}; "
      "install it with '${Python3_EXECUTABLE} -m pip install pybind11'")
  endif()
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKEDIR}")
  pybind11_add_module(_bowg python/bowg_pybind.cpp)
  target_link_libraries(_bowg PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS})
  target_include_directories(_bowg PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    include/BoWG
    include/DVision)
  set_target_properties(_bowg PROPERTIES
    CXX_STANDARD 11
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
  install(TARGETS _bowg LIBRARY DESTINATION python)
endif()

if(BUILD_DETECTOR)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
endif(BUILD_DETECTOR)

if(BUILD_TESTING AND BUILD_BOWG)
  add_executable(test_bowg_facade ../../tests/cpp/test_bowg_facade.cpp)
  target_link_libraries(test_bowg_facade ${PROJECT_NAME} ${OpenCV_LIBS})
  target_compile_definitions(test_bowg_facade PRIVATE
    BOWG_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  set_target_properties(test_bowg_facade PROPERTIES CXX_STANDARD 11)
  add_test(NAME bowg_facade COMMAND test_bowg_facade)
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_BOWG)
  install(DIRECTORY include/BoWG DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  install(DIRECTORY include/DUtils DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  install(DIRECTORY include/DVision DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
  install(FILES include/VocabularyBinary.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
endif()

configure_file(src/BoWG.cmake.in
  "${PROJECT_BINARY_DIR}/BoWGConfig.cmake" @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/BoWGConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES "${PROJECT_BINARY_DIR}/BoWGConfig.cmake" 
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/BoWG/)
install(DIRECTORY ${DEPENDENCY_INSTALL_DIR}/ DESTINATION ${CMAKE_INSTALL_PREFIX} OPTIONAL)
