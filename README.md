Hierarchical IMage REPresentation
======


## Installation

##### Dependencies
- [YARP](https://github.com/robotology/yarp)
- [iCub](https://github.com/robotology/icub-main)
- [icub-contrib-common](https://github.com/robotology/icub-contrib-common)
- [OpenCV](http://opencv.org/downloads.html)
- [SiftGPU](http://cs.unc.edu/~ccwu/siftgpu)

The `liblinear-1.91` package is a dependency for the `linearClassifierModule` and should be thus compiled beforehand. The resulting static library must be placed in the same directory of the `liblinear` package to then provide the environment variable **`LIBSVMLIN_DIR`** pointing to that location.

## Documentation

This repository contains a collection of modules that perform features extraction, coding, pooling and learning through a linear classifier.

Online documentation is available here: [http://robotology.github.com/himrep](http://robotology.github.com/himrep).

## License

Material included here is Copyright of _iCub Facility - Istituto Italiano di Tecnologia_ and is released under the terms of the GPL v2.0 or later. See the file LICENSE for details.
