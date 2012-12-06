coursera-heterogeneous
======================

Provides the `wb.h` header file for the [Heterogenous Parallel Programming](https://www.coursera.org/course/hetero) course from [Coursera](https://www.coursera.org/). This file can be used to work offline on the course assignments, provided you have access to CUDA hardware.

Running on OSX via Xcode
---------------

Make sure you have XCode & Cmake.
You can get cmake from macports via: `sudo port install cmake` or homebrew via:  `brew install cmake`

-  Download & Install NVIDIA CUDA: http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.36_macos.pkg
-  Clone repo and run: `cmake CMakeLists.txt -G Xcode`
-  Open the resulting Project.xcodeproj
-  Change Loading flags:
 -  Highlight Project
 -  Select mp0 under targets
 -  Under Build Settings, select Linking: Other Linker flags
 -  Double click to expand the flags
 -  Click the (+) at the bottom of the flags window
 -  6 Add the following: `-F/Library/Frameworks -framework CUDA`
- Now you can run & add debug points


