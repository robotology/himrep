
#ifndef __SIFT_GPU_EXTRACTOR_H__
#define __SIFT_GPU_EXTRACTOR_H__

#include <opencv2/opencv.hpp>

#if defined(__APPLE__)
   #include <OpenGL/gl.h>
   #include <OpenGL/glu.h>
#else
   #include "GL/gl.h"
#endif

#if !defined(SIFTGPU_STATIC) && !defined(SIFTGPU_DLL_RUNTIME)
// SIFTGPU_STATIC comes from compiler
#define SIFTGPU_DLL_RUNTIME
// Load at runtime if the above macro defined
// comment the macro above to use static linking
#endif


#ifdef SIFTGPU_DLL_RUNTIME
   #include <dlfcn.h>
   #define FREE_MYLIB dlclose
   #define GET_MYPROC dlsym
#endif

#include "SiftGPU.h"

#include <vector>
using namespace std;

class SiftGPU_Extractor
{
private:
   ComboSiftGPU                        *combo;
   SiftMatchGPU                        *matcher;
   SiftGPU                             *sift;

   vector<SiftGPU::SiftKeypoint>       keypoints_grid;

public:
    SiftGPU_Extractor();
        
    int getFeatureNum();
    
    bool setDenseGrid(int width, int height, int step, int scale);
    bool setDenseGrid(IplImage *img, int step, int scale);

    bool extractSift(IplImage *img, vector<SiftGPU::SiftKeypoint> *keypoints=NULL, vector<float> *descriptors=NULL, int feature_size=128);
    bool extractDenseSift(IplImage *img, vector<SiftGPU::SiftKeypoint> *keypoints=NULL, vector<float> *descriptors=NULL, int feature_size=128);
    
    bool getFeatureVector(vector<SiftGPU::SiftKeypoint> *keypoints, vector<float> *descriptors, int feature_size=128);
};



#endif



