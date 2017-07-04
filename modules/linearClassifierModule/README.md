## Installation

### Dependencies

This module needs [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/). We have included a version (source code) of the library here, in the `liblinear-1.91` folder.

In order to build it, you have to:

- compile from within the directory where the corresponding `cmakefile` is located

- place the resulting static library in the same directory of the `liblinear` package

- finally, set the environment variable `LIBSVMLIN_DIR` to point at the same location

### Compilation

Once the library has been built as explained above, you can compile this module just by setting the `BUILD_linearClassifierModule` flag to `ON` as explained [here](https://www.github.com/robotology/himrep/README/#compilation).

