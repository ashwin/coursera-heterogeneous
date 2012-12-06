coursera-heterogeneous
======================
 * This header is meant to be fix for compilation on local machine 
 for the course https://www.coursera.org/course/hetero.
 * I add Linux functionality and Makefile to original Windows version from https://bitbucket.org/ashwin/cudatimer
 * I busttled wbLog with numerous template variants so it fits all logs in mp0 and mp1.
 * I *added timer functionality* for Linux.
 * I splitted the Windows and Linux specific code into two version:
    * windows/cudatimer.cpp # set up your build system for windows *UNTESTED*
    * linux/cudatimer.cpp   # use make and makefile in test/Makefile for testing on mp0.cu
 * Headers wb.h and cudatimer.h are located in include directory.
 
