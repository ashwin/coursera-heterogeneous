CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(BUILD_SHARED_LIBS ON)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_EXE_LINKER_FLAGS "-lrt")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux") 

INCLUDE_DIRECTORIES(./include)

CUDA_ADD_EXECUTABLE(mp0 
    src/mp0.cc)

CUDA_ADD_EXECUTABLE(mp1 
    src/mp1.cu)

CUDA_ADD_EXECUTABLE(mp2 
    src/mp2.cu)

CUDA_ADD_EXECUTABLE(mp3 
    src/mp3.cu)

CUDA_ADD_EXECUTABLE(mp4
    src/mp4.cu)

CUDA_ADD_EXECUTABLE(mp5
    src/mp5.cu)

# Added tests, to execute use 'ctest'
enable_testing()

# tests for mp0
add_test(mp0 mp0)
set_tests_properties (mp0
        PROPERTIES FAIL_REGULAR_EXPRESSION "No CUDA GPU has been detected")

# sets the number of data sets to test
set (numtests 6)

# tests for mp1
set (i 0)
WHILE (${i} LESS ${numtests})
      add_test(mp1-${i} mp1 
        tests/mp1_data/${i}/input0.raw 
        tests/mp1_data/${i}/input1.raw 
        tests/mp1_data/${i}/output.raw)
      set_tests_properties (mp1-${i}
        PROPERTIES PASS_REGULAR_EXPRESSION "All tests passed")
      MATH(EXPR i "${i} + 1")
ENDWHILE (${i} LESS ${numtests} )

# tests for mp2
set (i 0)
WHILE (${i} LESS ${numtests})
      add_test(mp2-${i} mp2 
        tests/mp2_data/${i}/input0.raw 
        tests/mp2_data/${i}/input1.raw 
        tests/mp2_data/${i}/output.raw)
      set_tests_properties (mp2-${i}
        PROPERTIES PASS_REGULAR_EXPRESSION "All tests passed")
      MATH(EXPR i "${i} + 1")
ENDWHILE (${i} LESS ${numtests} )

# tests for mp3
set (i 0)
WHILE (${i} LESS ${numtests})
      add_test(mp3-${i} mp3 
        tests/mp3_data/${i}/input0.raw 
        tests/mp3_data/${i}/input1.raw 
        tests/mp3_data/${i}/output.raw)
      set_tests_properties (mp3-${i}
        PROPERTIES PASS_REGULAR_EXPRESSION "All tests passed")
      MATH(EXPR i "${i} + 1")
ENDWHILE (${i} LESS ${numtests})

# tests for mp4
set (i 0)
WHILE (${i} LESS ${numtests})
      add_test(mp4-${i} mp4 
        tests/mp4_data/${i}/input.raw
        tests/mp4_data/${i}/output.raw)
      set_tests_properties (mp4-${i}
        PROPERTIES PASS_REGULAR_EXPRESSION "All tests passed")
      MATH(EXPR i "${i} + 1")
ENDWHILE (${i} LESS ${numtests})
