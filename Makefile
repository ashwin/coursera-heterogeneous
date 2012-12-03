# Author:   Ondrej Platek, Copyright 2012, code is without any warranty!
# Created:  11:52:17 03/12/2012
# Modified: 11:52:17 03/12/2012

# mp0.exe: mp0.o cudatimer.o include/wb.h include/cudatimer.h
#     g++ -Iinclude -o $@ mp0.o cudatimer.o
CUDA=/opt/cuda

mp0.exe: mp0.o cudatimer.o
	g++ -o $@ -Iinclude -lcuda -lrt mp0.o cudatimer.o

mp0.o: test/mp0.cu
	nvcc -Iinclude  -c test/mp0.cu

cudatimer.o: include/cudatimer.h linux/cudatimer.cpp
	g++ -c -Iinclude -lrt $<
