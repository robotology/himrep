
#include "GIEFeatExtractor.h"

#include <sstream>

// Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
bool GIEFeatExtractor::cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)) )
		return false;

	if( CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)) )
		return false;

	memset(*cpuPtr, 0, size);
	std::cout << "cudaAllocMapped : " << size << " bytes" << std::endl;
	return true;
}

bool GIEFeatExtractor::cudaFreeMapped(void *cpuPtr)
{
    if ( CUDA_FAILED( cudaFreeHost(cpuPtr) ) )
        return false;
    std::cout << "cudaFreeMapped: OK" << std::endl;
}

bool GIEFeatExtractor::caffeToGIEModel( const std::string& deployFile,				// name for caffe prototxt
					                    const std::string& modelFile,				// name for model 
					                    const std::vector<std::string>& outputs,    // network outputs
					                    unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					                    std::ostream& gieModelStream)				// output stream for the GIE model
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	// parse the caffe model to populate the network, then set the outputs
	nvcaffeparser1::CaffeParser* parser = new nvcaffeparser1::CaffeParser;

	const bool useFp16 = builder->plaformHasFastFp16();
	std::cout << "Platform FP16 support: " << useFp16 << std::endl;
	std::cout << "Loading: " << deployFile << ", " << modelFile << std::endl;
	
	nvinfer1::DataType modelDataType = useFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
		parser->parse(deployFile.c_str(),		// caffe deploy file
					  modelFile.c_str(),		// caffe model file
					 *network,					// network definition that the parser will populate
					  modelDataType);

	if( !blobNameToTensor )
	{
		std::cout << "Failed to parse caffe network." << std::endl;
		return false;
	}
	
	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	const size_t num_outputs = outputs.size();
	
	for( size_t n=0; n < num_outputs; n++ )
		network->markOutput(*blobNameToTensor->find(outputs[n].c_str()));

	// Build the engine
	std::cout << "Configuring CUDA engine..." << std::endl;
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format, only on DriveCX
	if (useFp16)
		builder->setHalf2Mode(true);

	std::cout << "Building CUDA engine..." << std::endl;
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		std::cout << "Failed to build CUDA engine." << std::endl;
		return false;
	}

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	delete parser;

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	
	return true;
}

GIEFeatExtractor::GIEFeatExtractor(string _caffemodel_file,
            string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name,
            bool _timing ) {

    mEngine  = NULL;
	mInfer   = NULL;
	mContext = NULL;
	
    meanR = -1;
    meanG = -1;
    meanB = -1;
    resizeDims.n = -1;
    resizeDims.c = -1;
    resizeDims.w = -1;
    resizeDims.h = -1;

	mWidth     = 0;
	mHeight    = 0;
	mInputSize = 0;

	mInputCPU  = NULL;
	mInputCUDA = NULL;
	
	mOutputSize    = 0;
	mOutputDims = 0;

	mOutputCPU     = NULL;
	mOutputCUDA    = NULL;

    prototxt_file = "";
    caffemodel_file = "";
    blob_name = "";
    binaryproto_meanfile = "";

    timing = false;
    

    if( !init(_caffemodel_file, _binaryproto_meanfile, _meanR, _meanG, _meanB, _prototxt_file, _resizeWidth, _resizeHeight,  _blob_name ) )
	{
		std::cout << "GIEFeatExtractor: init() failed." << std::endl;
	}
	
    // Initialize timing flag
    timing = _timing;

}

bool GIEFeatExtractor::init(string _caffemodel_file, string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, string _prototxt_file, int _resizeWidth, int _resizeHeight, string _blob_name)
{
	
    cudaDeviceProp prop;
    int whichDevice;

    if ( CUDA_FAILED( cudaGetDevice(&whichDevice)) )
       return -1;
 
    if ( CUDA_FAILED( cudaGetDeviceProperties(&prop, whichDevice)) )
       return -1;

    if (prop.canMapHostMemory != 1)
    {
        std::cout << "Device cannot map memory!" << std::endl;
        return -1;
    }

    if ( CUDA_FAILED( cudaSetDeviceFlags(cudaDeviceMapHost)) )
        return -1;

    // Assign specified .caffemodel, .binaryproto, .prototxt files     
    caffemodel_file  = _caffemodel_file;
	binaryproto_meanfile = _binaryproto_meanfile;
    meanR = _meanR;
    meanG = _meanG;
    meanB = _meanB;
    prototxt_file = _prototxt_file;
    
    //Assign blob to be extracted
    blob_name = _blob_name;
    
    // Load and convert model
    std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

    if( !caffeToGIEModel( prototxt_file, caffemodel_file, std::vector< std::string > {  blob_name }, 1, gieModelStream) )
	{
		std::cout << "Failed to load: " << caffemodel_file << std::endl;
	}

	std::cout << caffemodel_file << ": loaded." << std::endl;

	// Create runtime inference engine execution context
	nvinfer1::IRuntime* infer = createInferRuntime(gLogger);
	if( !infer )
	{
		std::cout << "Failed to create InferRuntime." << std::endl;
	}
	
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);
	if( !engine )
	{
		std::cout << "Failed to create CUDA engine." << std::endl;
	}
	
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	if( !context )
	{
		std::cout << "failed to create execution context." << std::endl;
	}

	std::cout << "CUDA engine context initialized with " << engine->getNbBindings() << " bindings." << std::endl;
	
	mInfer   = infer;
	mEngine  = engine;
	mContext = context;

	// Determine dimensions of network bindings
	const int inputIndex  = engine->getBindingIndex("data");
	const int outputIndex = engine->getBindingIndex( blob_name.c_str() );

	std::cout << caffemodel_file << " input  binding index: " << inputIndex << std::endl;
	std::cout << caffemodel_file << " output binding index: " << outputIndex << std::endl;
	
	nvinfer1::Dims3 inputDims  = engine->getBindingDimensions(inputIndex);
	nvinfer1::Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	
	size_t inputSize  = inputDims.c * inputDims.h * inputDims.w * sizeof(float);
	size_t outputSize = outputDims.c * outputDims.h * outputDims.w * sizeof(float);
	
	std::cout << caffemodel_file << "input  dims (c=" << inputDims.c << " h=" << inputDims.h << " w=" << inputDims.w << ") size=" << inputSize << std::endl;
	std::cout << caffemodel_file << "output dims (c=" << outputDims.c << " h=" << outputDims.h << " w=" << outputDims.w << ") size=" << outputSize << std::endl;
	
	// Allocate memory to hold the input image
	if ( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		std::cout << "Failed to alloc CUDA mapped memory for input, " << inputSize << " bytes" << std::endl;
	}
	mInputSize   = inputSize;
	mWidth       = inputDims.w;
	mHeight      = inputDims.h;
	
	// Allocate output memory to hold the result
	if( !cudaAllocMapped((void**)&mOutputCPU, (void**)&mOutputCUDA, outputSize) )
	{
		std::cout << "Failed to alloc CUDA mapped memory for output, " << outputSize << " bytes" << std::endl;
	}
	mOutputSize    = outputSize;
	mOutputDims    = outputDims.c;
	
	std::cout << caffemodel_file << ": initialized." << std::endl;

    // Mean image initialization
    if (binaryproto_meanfile!="")
    {
	    nvcaffeparser1::IBinaryProtoBlob* meanBlob = nvcaffeparser1::CaffeParser::parseBinaryProto(binaryproto_meanfile.c_str());
        resizeDims = meanBlob->getDimensions();
        const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());  // expected to be float* (c,h,w)
        float *meanDataChangeable = (float *) malloc(resizeDims.w*resizeDims.h*resizeDims.c*resizeDims.n*sizeof(float));
        memcpy(meanDataChangeable, meanData, resizeDims.w*resizeDims.h*resizeDims.c*resizeDims.n*sizeof(float) );
        
        meanBlob->destroy();

        cv::Mat tmpMat(resizeDims.h, resizeDims.w, CV_32FC3, meanDataChangeable);
        tmpMat.copyTo(meanMat);

        free(meanDataChangeable);
        
    }
    else
    {
        resizeDims.h = _resizeHeight;
        resizeDims.w = _resizeWidth;
        resizeDims.c = 3;
        resizeDims.n = 1;
    }

	return true;
}

GIEFeatExtractor::~GIEFeatExtractor()
{
	if( mEngine != NULL )
	{
		mEngine->destroy();
		mEngine = NULL;
	}
		
	if( mInfer != NULL )
	{
		mInfer->destroy();
		mInfer = NULL;
	}

    cudaFreeMapped(mOutputCPU);
    cudaFreeMapped(mInputCPU);
}

bool GIEFeatExtractor::extract_singleFeat_1D(cv::Mat &imMat, vector<float> &features, float (&times)[2])
{

    // Check input image 
    if (imMat.empty())
	{
		std::cout << "GIEFeatExtractor::extract_singleFeat_1D(): empty imMat!" << std::endl;
		return -1;
	}

    times[0] = 0.0f;
    times[1] = 0.0f;

    // Start timing
    cudaEvent_t startPrep, stopPrep, startNet, stopNet;
    if (timing)
    {
        cudaEventCreate(&startPrep);
        cudaEventCreate(&startNet);
        cudaEventCreate(&stopPrep);
        cudaEventCreate(&stopNet);
        cudaEventRecord(startPrep, NULL);
        cudaEventRecord(startNet, NULL);
    }

    // Image preprocessing

    // convert to float (with range 0-255)
    imMat.convertTo(imMat, CV_32FC3);
    
    // resize 
    if (imMat.rows != resizeDims.h || imMat.cols != resizeDims.w)
    {
        if (imMat.rows > resizeDims.h || imMat.cols > resizeDims.w)
        {
            cv::resize(imMat, imMat, cv::Size(resizeDims.h, resizeDims.w), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(imMat, imMat, cv::Size(resizeDims.h, resizeDims.w), 0, 0, CV_INTER_LINEAR);
        }
    }

    // subtract mean 
    if (meanR==-1)
    {
        if (!meanMat.empty() && imMat.rows==meanMat.rows && imMat.cols==meanMat.cols && imMat.channels()==meanMat.channels() && imMat.type()==meanMat.type())
        {
            imMat = imMat - meanMat;
        }
        else
        {
            std::cout << "GIEFeatExtractor::extract_singleFeat_1D(): cannot subtract mean image!" << std::endl;
            return -1;
        }
    }
    else
    {  
        imMat = imMat - cv::Scalar(meanR, meanG, meanB);
    }

    // crop to input dimension (central crop)
    if (imMat.cols>=mWidth && imMat.rows>=mHeight)
    {
        cv::Rect imROI(floor((imMat.cols-mWidth)*0.5f), floor((imMat.rows-mHeight)*0.5f), mWidth, mHeight);
        imMat(imROI).copyTo(imMat);
    } else
    {
        cv::resize(imMat, imMat, cv::Size(mHeight, mWidth), 0, 0, CV_INTER_LINEAR);
    }

     if ( !imMat.isContinuous() ) 
          imMat = imMat.clone();

    // copy 
    CUDA( cudaMemcpy(mInputCPU, imMat.data, mInputSize, cudaMemcpyDefault) );
    // memcpy(mInputCPU, imMat.data);

	void* inferenceBuffers[] = { mInputCUDA, mOutputCUDA };

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopPrep, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopPrep);

        cudaEventElapsedTime(times, startPrep, stopPrep);
    }

	mContext->execute(1, inferenceBuffers);
	//CUDA(cudaDeviceSynchronize());

    features.insert(features.end(), &mOutputCPU[0], &mOutputCPU[mOutputDims]);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopNet, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopNet);

        cudaEventElapsedTime(times+1, startNet, stopNet);

    }

    return 1;

}

