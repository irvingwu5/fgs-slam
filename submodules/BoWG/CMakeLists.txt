# 文件作用：配置 BoWG 编译目标、依赖和安装导出；隶属于构建配置模块。
cmake_minimum_required(VERSION 3.0)
project(BoWG)

include(ExternalProject)
option(BUILD_BOWG "Build BoWG" ON)  
option(BUILD_DETECTOR "Build detector application" ON)

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

  # Train Vocabulary
  src/Train_Voc/train_voc.cpp
  
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

if(BUILD_DETECTOR)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 11)
endif(BUILD_DETECTOR)

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
