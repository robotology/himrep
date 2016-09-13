#ifndef GIEFEATEXTRACTOR_H_
#define GIEFEATEXTRACTOR_H_

#include <string>
#include <math.h>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// GIE
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "cudaUtility.h"

using namespace cv;

// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
    void log( Severity severity, const char* msg ) override
    {
    if( severity != Severity::kINFO )
        std::cout << msg << std::endl;
    }
};

class GIEFeatExtractor {

protected:
    
    bool cudaFreeMapped(void *cpuPtr);

    bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size );

    bool caffeToGIEModel( const std::string& deployFile,		    // name for caffe prototxt
					      const std::string& modelFile,	            // name for model
                          const std::string& binaryprotoFile,       // name for .binaryproto
					      const std::vector<std::string>& outputs,  // network outputs
					      unsigned int maxBatchSize,		        // batch size - NB must be at least as large as the batch we want to run with)
					      std::ostream& gieModelStream);		    // output stream for the GIE model

    bool init(string _caffemodel_file,
            string _binaryproto_meanfile, float meanR, float meanG, float meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name);

    nvinfer1::IRuntime* mInfer;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;

    //cv::Mat meanMat;
    float *meanData;
    vector<float> mean_values;

    nvinfer1::Dims4 resizeDims;

    uint32_t mWidth;
    uint32_t mHeight;
    uint32_t mInputSize;
    float*   mInputCPU;
    float*   mInputCUDA;
	
    uint32_t mOutputSize;
    uint32_t mOutputDims;
    float*   mOutputCPU;
    float*   mOutputCUDA;

    Logger  gLogger;

public:

    string prototxt_file;
    string caffemodel_file;
    string blob_name;
    string binaryproto_meanfile;

    bool timing;
    
    GIEFeatExtractor(string _caffemodel_file,
            string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name,
            bool _timing );

    ~GIEFeatExtractor();

    bool extract_singleFeat_1D(cv::Mat &image, vector<float> &features,  float (&times)[2]);
};

#endif /* GIEFEATEXTRACTOR_H_ */
