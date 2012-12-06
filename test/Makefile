# set the path to main root of this repo 
WBH=..
INC=-I${WBH}/include  # wb.h and cudatimer.h is in include folder of this repo

%: %.cu 
	nvcc -o $@ -lrt ${INC} ${WBH}/linux/cudatimer.cpp $<

tests: test_mp0

test_mp0: mp0
	echo "No input needed, running ./mp0"
	./mp0

.PHONY: clean test_mp1 test

clean:
	rm mp[0-9]
