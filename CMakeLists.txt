cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(libwb)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(APPLE)
  set(CMAKE_EXE_LINKER_FLAGS "-lm -F/Library/Frameworks -framework CUDA")
  set(CUDA_HOST_COMPILER "/usr/bin/gcc")
endif(APPLE)

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



# CTEST TESTING
#
# To test all MPs against the official datasets type 'ctest -V' at the
# terminal. To test a specific MP type 'ctest -L mpX -V'
enable_testing()

add_test(
    NAME mp0
    COMMAND mp0
)
set_tests_properties(
    mp0
    PROPERTIES FAIL_REGULAR_EXPRESSION "No CUDA GPU has been detected"
    LABELS mp0
)

set(DATASETS 9)

set(TESTS mp1 mp2 mp3)
foreach (TEST ${TESTS})
    set(PATH tests/${TEST})
    foreach (i RANGE 0 ${DATASETS})
        set(TESTLABEL ${TEST}-${i})
        add_test(
            NAME ${TESTLABEL}
            COMMAND ${TEST}
            ${PATH}/${i}/input0.raw
            ${PATH}/${i}/input1.raw
            ${PATH}/${i}/output.raw
        )
        set_tests_properties(
            ${TESTLABEL}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${TESTLABEL}
        )
    endforeach ()
endforeach ()

set(TEST mp4)
set(PATH tests/${TEST})
foreach (i RANGE 0 ${DATASETS})
    set(TESTLABEL ${TEST}-${i})
    add_test(
        NAME ${TESTLABEL}
        COMMAND ${TEST}
        ${PATH}/${i}/input0.ppm
        ${PATH}/${i}/input1.csv
        ${PATH}/${i}/output.ppm
    )
    set_tests_properties(
        ${TESTLABEL}
        PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
        LABELS ${TESTLABEL}
    )
endforeach ()

set(TESTS mp5 mp6)
foreach (TEST ${TESTS})
    set(PATH tests/${TEST})
    if (${TEST} STREQUAL "mp5")
        set(INPUT_FILE "input0")
    else ()
        set(INPUT_FILE "input")
    endif ()
    foreach (i RANGE 0 ${DATASETS})
        set(TESTLABEL ${TEST}-${i})
        add_test(
            NAME ${TESTLABEL}
            COMMAND ${TEST}
            ${PATH}/${i}/${INPUT_FILE}.raw
            ${PATH}/${i}/output.raw
        )
        set_tests_properties(
            ${TESTLABEL}
            PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
            LABELS ${TESTLABEL}
        )
    endforeach ()
endforeach ()
