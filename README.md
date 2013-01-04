About
=====
***

This **coursera-heterogeneous** project provides offline resources to work on the assignments of [**Heterogenous Parallel Programming**](https://www.coursera.org/course/hetero) course from [**Coursera**](https://www.coursera.org/).
This is a collaborative effort by the students of the course and you are welcome to [contribute](#contributors) to improve it.

Files available include:

- WB API header file (`wb.h`)
- Generate data to test assignments (`GenDataMP1.cpp`, `GenDataMP2.cpp`, ...)
- Assignment files (`mp0.cu`, `mp1.cu`, ...)

All of this works **only** if you have access to CUDA hardware.

Usage
=====
***

On Windows with Visual Studio
-----------------------------

- Update NVIDIA driver for your CUDA hardware
- Download and install Visual Studio or Visual C++ Express Edition
- Download and install the [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- Create a [CUDA project in Visual Studio](http://google.com/search?q=cuda%20project%20in%20visual%20studio)
- Place this `wb.h` header in same directory as your CUDA source file (`mp1.cu` for example)
- Compile and run!

On OS X with Xcode
------------------

Make sure you have XCode (and/or the Command Line Tools) and Cmake installed.
You can get cmake from macports via: `sudo port install cmake` or homebrew via:  `brew install cmake`

### Dependencies

- If you wish to use Xcode as your IDE, download and install the latest version from the [Mac App Store](https://itunes.apple.com/au/app/xcode/id497799835?mt=12)
- Download and install the [Command Line Tools](https://developer.apple.com/downloads) from Apple's developer website or, if you are using Xcode, via the Downloads tab in Xcode's Preferences
- Download and install the [NVIDIA CUDA toolkit](http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.36_macos.pkg)
- Install cmake from either macports (`sudo port install cmake`) or homebrew (`brew install cmake`)
- Clone the coursera-heterogenous repository
- At this point, you can use either Xcode or the traditional Unix command line

### Using Xcode

Initial setup:

- Though Xcode can build and run CUDA projects, it does not integrate NVIDIA's cuda-gdb debugger into the IDE. OpenCL projects, in contrast, have complete integration with Xcode
- If you haven't done so already, you need to install the [Command Line Tools](https://developer.apple.com/downloads) so that the UNIX development environment is exposed to NVIDIA's CUDA compiler `nvcc`
- From the repository's root directory type: `cmake CMakeLists.txt -G Xcode`
- Open the resulting Project.xcodeproj

To run each assignment:

- You will need to edit each assignment's Scheme through the Product, Edit Scheme menu
 - Select the Run build action settings and, under the Arguments tab, enter the arguments that will be passed to the program when it launches
 - Make sure that, for each dataset file, you enter that file's absolute path
- Compile and run!

### Using the Unix command line

Initial setup:

- From the repository's root directory type: `cmake CMakeLists.txt`

To run each assignment:

- To compile an assignment, type `make mp<N>`, where `<N>` is the assignment's number
 - For example, to compile the MP2 assignment type `make mp2`
- To run an assignment, type `./mp<N> tests/mp<N>_data/<D>/*`, where `<N>` is a homework number and `<D>` is a dataset number
 - For example, to execute assignment MP2 on dataset 0 type `./mp3 tests/mp3_data/0/*`

<a name="contributors"/>
Contributors
============
***

[View list of contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)

We welcome improvements to this code. Fork it, make your change and give me a pull request. Please follow the coding conventions already in use in the source files.


License
=======
***

All the files in this project are shared under the [MIT License](http://opensource.org/licenses/mit-license.php).

Copyright (C) 2012 [Contributors](https://github.com/ashwin/coursera-heterogeneous/contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.