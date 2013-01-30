CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(BUILD_SHARED_LIBS ON)
if(UNIX AND NOT APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-lm -lrt")
endif(UNIX AND NOT APPLE)
if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-lm -F/Library/Frameworks -framework CUDA")
  set(CUDA_HOST_COMPILER "/usr/bin/gcc")
endif(APPLE)

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

CUDA_ADD_EXECUTABLE(mp6
    src/mp6.cu)

# Added tests, to execute use 'ctest'
enable_testing()

# tests for mp0
add_test(mp0 mp0)
set_tests_properties (mp0
        PROPERTIES FAIL_REGULAR_EXPRESSION "No CUDA GPU has been detected"
        LABELS mp0)

set (numtests 6)

# tests for mp1 mp2 mp3
set (tests mp1 mp2 mp3)
FOREACH (test ${tests})
    set (path tests/${test}_data)
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}_data-${i})
          add_test(${testname} ${test} 
            ${path}/${i}/input0.raw 
            ${path}/${i}/input1.raw 
            ${path}/${i}/output.raw)
          set_tests_properties (${testname}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${testname})
          MATH(EXPR i "${i} + 1")
    ENDWHILE (${i} LESS ${numtests} )
ENDFOREACH(test)

# tests for mp4 mp5
set (tests mp4 mp5)
set (i 0)
FOREACH (test ${tests})
    set (path tests/${test}_data)
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}_data-${i})
          add_test(${testname} ${test} 
            ${path}/${i}/input.raw
            ${path}/${i}/output.raw)
          set_tests_properties (${testname}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${testname})
          MATH(EXPR i "${i} + 1")
    ENDWHILE (${i} LESS ${numtests})
ENDFOREACH(test)

# tests for mp6
set (tests mp6)
FOREACH (test ${tests})
    set (path tests/${test}_data)
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}_data-${i})
          add_test(${testname} ${test} 
            ${path}/${i}/input0.ppm 
            ${path}/${i}/input1.raw 
            ${path}/${i}/output.ppm)
          set_tests_properties (${testname}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${testname})
          MATH(EXPR i "${i} + 1")
    ENDWHILE (${i} LESS ${numtests} )
ENDFOREACH(test)


# official graded tests
set (numtests 10)

# tests for mp1 mp2 mp3
set (tests mp1 mp2 mp3)
FOREACH (test ${tests})
    set (path tests/${test}_grade)
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}_grade-${i})
          add_test(${testname} ${test} 
            ${path}/${i}/input0.raw 
            ${path}/${i}/input1.raw 
            ${path}/${i}/output.raw)
          set_tests_properties (${testname}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${testname})
          MATH(EXPR i "${i} + 1")
    ENDWHILE (${i} LESS ${numtests} )
ENDFOREACH(test)
