About
=====
***

This **coursera-heterogeneous** project provides offline resources to work on the assignments of [**Heterogenous Parallel Programming**](https://www.coursera.org/course/hetero) course from [**Coursera**](https://www.coursera.org/).
This is a collaborative effort by the students of the course and you are welcome to [contribute](#contributors) to improve it.

Files available include:

- WB API header file (`wb.h`)
- Generate data to test assignments (`GenDataMP1.cpp`, `GenDataMP2.cpp`, ...)
- Assignment files (`MP0.cu`, `MP1.cu`, ...)

All of this works **only** if you have access to CUDA hardware.

Usage
=====
***

On Windows with Visual Studio
-----------------------------

- Update NVIDIA driver for your CUDA hardware
- Download and install Visual Studio or Visual C++ Express Edition.
- Download and install [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- Create [CUDA project in Visual Studio](http://google.com/search?q=cuda%20project%20in%20visual%20studio)
- Place this `wb.h` in same directory as your CUDA source file (`mp1.cu` for example)
- Compile and run!

On OS X with Xcode
------------------

Make sure you have XCode & Cmake.
You can get cmake from macports via: `sudo port install cmake` or homebrew via:  `brew install cmake`

### Dependencies

- Download and install Xcode from the Apple App Store.
- Download and install NVIDIA CUDA for MacOS from [here](http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.36_macos.pkg)
- Install cmake from macports (sudo port install cmake) or homebrew (brew install cmake).
- Clone the coursera-heterogenous repository.
- At this point, you can either use the Xcode GUI or use a traditional Unix makefile.

### Using Xcode GUI

Initial setup:

- cd into the repository root directory and run: `cmake CMakeLists.txt -G Xcode`
-  Open the resulting Project.xcodeproj: `open Project.xcodeproj`
-  Change Loading flags:
 -  Highlight Project
 -  Select mp0 under targets
 -  Under Build Settings, select Linking: Other Linker flags
 -  Make sure that the disclosure arrow for "Other Linker Flags" is pointing to the right (i.e. closed)
 -  Double click to expand the flags
 -  Click the (+) at the bottom of the flags window
 -  Add the following: `-F/Library/Frameworks -framework CUDA`
 -  Repeat for other targets: mp1, m2, mp3, etc.

To run each homework assignment:

- Now you can run & add debug points

### Using Unix makefile

Initial setup:

- cd into the repository root directory and run: `cmake CMakeLists.txt`

To run each homework assignment:

- To compile a homework program, run `make mp<N>`, where <N> is a homework number.
 - For example, to compile MP2, run `make mp2`.
- After compiling, to run the homework program, run `./mp<N> tests/mp<N>_data/<D>/*`, where <N> is a homework number and <D> is a dataset number.
 - For example, to execute MP2 on dataset 0, run `./mp3 tests/mp3_data/0/*`.

You can also test each homework (`mp`) against the official datasets
using the command `ctest` (part of cmake):

- To run all the `mps` against the official datasets: `ctest`.
- To run a specific mp, for example MP2, run `ctest -L mp2`

The details of the execution of the tests can be found in the folder
`Testing/Temporary/LastTest.log`. Or can be viewed adding the `-V` or
`-VV` for extra verbose information, i.e. `ctest -L mp2 -V`.

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
