CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(BUILD_SHARED_LIBS ON)
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

# We do this painfully here because setting CMAKE_EXE_LINKER_FLAGS
# gives error on Ubuntu. Because CMake prepends the linker flags.
if (UNIX AND NOT APPLE)
    set(TIMER_FLAGS "-lm -lrt")
    target_link_libraries(mp0 ${TIMER_FLAGS})
    target_link_libraries(mp1 ${TIMER_FLAGS})
    target_link_libraries(mp2 ${TIMER_FLAGS})
    target_link_libraries(mp3 ${TIMER_FLAGS})
    target_link_libraries(mp4 ${TIMER_FLAGS})
    target_link_libraries(mp5 ${TIMER_FLAGS})
    target_link_libraries(mp6 ${TIMER_FLAGS})
endif(UNIX AND NOT APPLE)

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
    set (path tests/${test})
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}-${i})
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
    set (path tests/${test})
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}-${i})
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
    set (path tests/${test})
    set (i 0)
    WHILE (${i} LESS ${numtests})
          set(testname ${test}-${i})
          add_test(${testname} ${test}
            ${path}/${i}/input0.ppm
            ${path}/${i}/input1.csv
            ${path}/${i}/output.ppm)
          set_tests_properties (${testname}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${testname})
          MATH(EXPR i "${i} + 1")
    ENDWHILE (${i} LESS ${numtests} )
ENDFOREACH(test)
