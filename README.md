About
=====
***

This **coursera-heterogeneous** project provides offline resources to work on the assignments of [**Heterogenous Parallel Programming**](https://www.coursera.org/course/hetero) course from [**Coursera**](https://www.coursera.org/).
This is a collaborative effort by the students of the course and you are welcome to [contribute](#contributors) to improve it.

Files available include:

- WB API header file (`wb.h`)
- Official assignment files (`mp0.cu`, `mp1.cu`, ...)
- Official test datasets
- Programs to generate extra datasets to test assignments (`GenDataMP1.cpp`, `GenDataMP2.cpp`, ...)

All of this works **only** if you have access to CUDA hardware or an OpenCL installation.

Usage
=====
***

On Windows with Visual Studio
-----------------------------

### Dependencies

- Update the NVIDIA driver for your CUDA hardware
- Download and install Visual Studio or Visual C++ Express Edition
- Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads#win)

### Using Visual Studio

- Create a [CUDA project in Visual Studio](http://google.com/search?q=cuda%20project%20in%20visual%20studio)
- Place the `wb.h` header in the same directory as your CUDA source file (`mp1.cu` for example)
- Compile and run!

On OS X with Xcode
------------------

Make sure you have Xcode (and/or the Command Line Tools) and CMake installed (CMake is a cross-platform, open-source build system). The preferred method of obtaining CMake is from [Homebrew](http://brew.sh/) via: `brew update && brew install cmake`

### Dependencies

- If you wish to use Xcode as your IDE, download and install the latest version from [Apple's](https://developer.apple.com/xcode/downloads/) developer website
- Download and install the [Command Line Tools](https://developer.apple.com/downloads) from Apple's developer website
- Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads#mac)
- Install CMake from [Homebrew](http://brew.sh/) (`brew update && brew install cmake`)
- Clone the coursera-heterogenous repository
- At this point, you can either use Xcode (recommended only for the OpenCL assignments) or the traditional Unix command line (see the instructions below)

### Using Xcode

Initial setup:

- Though Xcode can build and run CUDA projects, it does not integrate NVIDIA's cuda-gdb debugger into the IDE. In contrast, OpenCL projects have complete integration with Xcode and LLDB  (with `CL_DEVICE_TYPE_CPU` selected during development)
- If you haven't done so already, you need to install the [Command Line Tools](https://developer.apple.com/downloads) so that the UNIX development environment is exposed to NVIDIA's CUDA compiler `nvcc`
- From the repository's root directory type: `cmake CMakeLists.txt -G Xcode`
- Open the resulting libwb.xcodeproj

To run each assignment:

- You will need to edit each assignment's Scheme through the Product, Edit Scheme menu
 - Select the Run build action settings and, under the Arguments tab, enter the arguments that will be passed to the program when it launches (make sure that you enter each data set file's absolute path)
- Compile and run!

On Unix with the command line
-----------------------------

### Dependencies

- Download and install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads#linux)
- Install CMake through your system's package manager
 - Ubuntu `sudo apt-get install cmake`
 - Fedora `sudo yum install cmake`
- Clone the coursera-heterogenous repository

### Using the command line

Initial setup:

- From the repository's root directory type: `cmake CMakeLists.txt`

To run each assignment:

- To compile an assignment, type: `make mp<N>`, where `<N>` is that assignment's number
 - For example, to compile the MP2 assignment type: `make mp2`
- To run an assignment, type: `./mp<N> tests/mp<N>/<D>/*`, where `<N>` is a homework number and `<D>` is a data set number
 - For example, to execute assignment MP2 on data set 0 type: `./mp2 tests/mp2/0/*`

You can also test each assignment against the official datasets
using CMake's `ctest`:

- To run all of the MP assignments against the official datasets, from the repository's root directory, type: `ctest -V`
- To run a specific MP assignment, for example MP2, type `ctest -L mp2 -V`

You can view all of the information presented during the testing phase in the folder `Testing/Temporary/LastTest.log`

<a name="contributors"/>
Contributors
============
***

[View list of contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)

We welcome improvements to this code. Simply fork it, make your change, and initiate  a pull request! (Please follow the coding conventions already in use in the source files).


License
=======
***

All the files in this project are shared under the [MIT License](http://opensource.org/licenses/mit-license.php).

Copyright (C) 2012 [Contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
