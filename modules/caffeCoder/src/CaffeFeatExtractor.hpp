#ifndef CAFFEFEATEXTRACTOR_H_
#define CAFFEFEATEXTRACTOR_H_

#include <string>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// Boost
#include "boost/algorithm/string.hpp"
#include "boost/make_shared.hpp"

// Caffe

#include "caffe-version.h"

#if (CAFFE_MAJOR >= 1)
    // new caffe headers
    #include "caffe/caffe.hpp"
    #include "caffe/layers/memory_data_layer.hpp"
#else
    // old caffe headers
    #include "caffe/blob.hpp"
    #include "caffe/common.hpp"
    #include "caffe/net.hpp"
    #include "caffe/proto/caffe.pb.h"
    #include "caffe/util/io.hpp"
    #include "caffe/vision_layers.hpp"
#endif

using namespace std;
using namespace caffe;

template<class Dtype>
class CaffeFeatExtractor {

    string caffemodel_file;
    string prototxt_file;

    caffe::shared_ptr<Net<Dtype> > feature_extraction_net;

    int mean_width;
    int mean_height;
    int mean_channels;

    vector<string> blob_names;

    bool gpu_mode;
    int device_id;

public:

    bool timing;

    CaffeFeatExtractor(string _caffemodel_file,
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_names,
            string _compute_mode,
            int _device_id,
            bool _timing_extraction);

    float extractBatch_multipleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features);

    float extractBatch_singleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features);

    float extractBatch_multipleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features);

    float extractBatch_singleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features);

    float extract_multipleFeat(cv::Mat &image, vector< Blob<Dtype>* > &features);

    float extract_singleFeat(cv::Mat &image, Blob<Dtype> *features);

    float extract_multipleFeat_1D(cv::Mat &image, vector< vector<Dtype> > &features);

    bool extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features, float (&times)[2]);
};

template <class Dtype>
CaffeFeatExtractor<Dtype>::CaffeFeatExtractor(string _caffemodel_file,
        string _prototxt_file, int _resizeWidth, int _resizeHeight,
        string _blob_names,
        string _compute_mode,
        int _device_id,
        bool _timing) {

    // Setup the GPU or the CPU mode for Caffe
    if (strcmp(_compute_mode.c_str(), "GPU") == 0 || strcmp(_compute_mode.c_str(), "gpu") == 0) {

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using GPU" << std::endl;

        gpu_mode = true;
        device_id = _device_id;

        Caffe::CheckDevice(device_id);

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using device_id = " << device_id << std::endl;

        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);

        // Optional: to check that the GPU is working properly...
        Caffe::DeviceQuery();

    } else
    {
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): using CPU" << std::endl;

        gpu_mode = false;
        device_id = -1;

        Caffe::set_mode(Caffe::CPU);
    }

    // Assign specified .caffemodel and .prototxt files
    caffemodel_file = _caffemodel_file;
    prototxt_file = _prototxt_file;

    // Network creation using the specified .prototxt
    feature_extraction_net = boost::make_shared<Net<Dtype> > (prototxt_file, caffe::TEST);

    // Network initialization using the specified .caffemodel
    feature_extraction_net->CopyTrainedLayersFrom(caffemodel_file);

    // Mean image initialization

    mean_width = 0;
    mean_height = 0;
    mean_channels = 0;

    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    TransformationParameter tp = memory_data_layer->layer_param().transform_param();

    if (tp.has_mean_file())
    {
        const string& mean_file = tp.mean_file();
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): loading mean file from " << mean_file << std::endl;

        BlobProto blob_proto;
        ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

        Blob<Dtype> data_mean;
        data_mean.FromProto(blob_proto);

        mean_channels = data_mean.channels();
        mean_width = data_mean.width();
        mean_height = data_mean.height();

    } else if (tp.mean_value_size()>0)
    {

        const int b = tp.mean_value(0);
        const int g = tp.mean_value(1);
        const int r = tp.mean_value(2);

        mean_channels = tp.mean_value_size();
        mean_width = _resizeWidth;
        mean_height = _resizeHeight;

        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): B " << b << "   G " << g << "   R " << r << std::endl;
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): resizing anysotropically to " << " W: " << mean_width << " H: " << mean_height << std::endl;
    }
    else
    {
        std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): Error: neither mean file nor mean value in prototxt!" << std::endl;
    }

    // Check that requested blobs exist
    boost::split(blob_names, _blob_names, boost::is_any_of(","));
    for (size_t i = 0; i < blob_names.size(); i++) {
        if (!feature_extraction_net->has_blob(blob_names[i]))
        {
            std::cout << "CaffeFeatExtractor::CaffeFeatExtractor(): unknown feature blob name " << blob_names[i] << " in the network " << prototxt_file << std::endl;
        }
    }

    // Initialize timing flag
    timing = _timing;

}


template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_multipleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features) {

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize labels to zero
    vector<int> labels(images.size(), 0);

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size

    if (memory_data_layer->batch_size()!=new_batch_size)
    {
        if (images.size()%new_batch_size==0)
        {
            memory_data_layer->set_batch_size(new_batch_size);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
        else
        {
            if (images.size()%memory_data_layer->batch_size()==0)
            {
                cout << "WARNING: image number is not multiple of requested batch size, leaving the old one..." << endl;
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            } else
            {
                cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)..." << endl;
                memory_data_layer->set_batch_size(1);
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            }

        }

    } else
    {
        if (images.size()%memory_data_layer->batch_size()!=0)
        {
            cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)..." << endl;
            memory_data_layer->set_batch_size(1);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
    }

    int num_batches = images.size()/new_batch_size;

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized anisotropically (BILINEAR)
    // if it is downsampled, LANCZOS4 is used for antialiasing

    for (int i=0; i<images.size(); i++)
    {
        if (images[i].rows != mean_height || images[i].cols != mean_height)
        {
            if (images[i].rows > mean_height || images[i].cols > mean_height)
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
            }
            else
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
            }
        }
    }

    memory_data_layer->AddMatVector(images,labels);

    size_t num_features = blob_names.size();

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results;

    for (int b=0; b<num_batches; b++)
    {
        results = feature_extraction_net->Forward();

        for (int i = 0; i < num_features; ++i) {

            const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[i]);

            int batch_size = feature_blob->num();
            int channels = feature_blob->channels();
            int width = feature_blob->width();
            int height = feature_blob->height();

            features.push_back(new Blob<Dtype>(batch_size, channels, height, width));

            features.back()->CopyFrom(*feature_blob);
        }

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        float msecPerImage = msecTotal/(float)images.size();

        return msecPerImage;
    }
    else
    {
        return 0;
    }

}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_singleFeat(vector<cv::Mat> &images, int new_batch_size, vector< Blob<Dtype>* > &features) {

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize the labels to zero
    vector<int> labels(images.size(), 0);

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size

    if (memory_data_layer->batch_size()!=new_batch_size)
    {
        if (images.size()%new_batch_size==0)
        {
            memory_data_layer->set_batch_size(new_batch_size);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
        else
        {
            if (images.size()%memory_data_layer->batch_size()==0)
            {
                cout << "WARNING: image number is not multiple of requested batch size, leaving the old one." << endl;
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            } else
            {
                cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
                memory_data_layer->set_batch_size(1);
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            }

        }

    } else
    {
        if (images.size()%memory_data_layer->batch_size()!=0)
        {
            cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
            memory_data_layer->set_batch_size(1);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
    }

    int num_batches = images.size()/new_batch_size;

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied

    for (int i=0; i<images.size(); i++)
    {
        if (images[i].rows != mean_height || images[i].cols != mean_height)
        {
            if (images[i].rows > mean_height || images[i].cols > mean_height)
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
            }
            else
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
            }
        }
    }

    memory_data_layer->AddMatVector(images,labels);

    size_t num_features = blob_names.size();
    if (num_features!=1)
    {
        cout<< "Error! The list of features to be extracted has not size one!" << endl;
        return -1;
    }

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results;

    for (int b=0; b<num_batches; b++)
    {
        results = feature_extraction_net->Forward();

        const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

        int batch_size = feature_blob->num();
        int channels = feature_blob->channels();
        int width = feature_blob->width();
        int height = feature_blob->height();

        features.push_back(new Blob<Dtype>(batch_size, channels, height, width));

        features.back()->CopyFrom(*feature_blob);

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        float msecPerImage = msecTotal/(float)images.size();

        return msecPerImage;
    }
    else
    {
        return 0;
    }
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_multipleFeat(cv::Mat &image, vector< Blob<Dtype>* > &features)
{

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize the labels to zero
    int label = 0;

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1

    if (memory_data_layer->batch_size()!=1)
    {
        memory_data_layer->set_batch_size(1);
        cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
    }

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied


    if (image.rows != mean_height || image.cols != mean_height)
    {
        if (image.rows > mean_height || image.cols > mean_height)
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
        }
    }

    memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

    size_t num_features = blob_names.size();

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results = feature_extraction_net->Forward();

    for (int f = 0; f < num_features; ++f) {

        const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

        int batch_size = feature_blob->num(); // should be 1
        if (batch_size!=1)
        {
            cout << "Error! Retrieved more than one feature, exiting..." << endl;
            return -1;
        }

        int channels = feature_blob->channels();
        int width = feature_blob->width();
        int height = feature_blob->height();

        features.push_back(new Blob<Dtype>(1, channels, height, width));

        features.back()->CopyFrom(*feature_blob);

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;

        cudaEventElapsedTime(&msecTotal, start, stop);

        return msecTotal;
    }
    else
    {
        return 0;
    }

}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_singleFeat(cv::Mat &image, Blob<Dtype> *features)
{

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize label to zero
    int label = 0;

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1

    if (memory_data_layer->batch_size()!=1)
    {
        memory_data_layer->set_batch_size(1);
        cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
    }

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied

    if (image.rows != mean_height || image.cols != mean_height)
    {
        if (image.rows > mean_height || image.cols > mean_height)
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
        }
    }

    memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

    size_t num_features = blob_names.size();
    if(num_features!=1)
    {
        cout<< "Error! The list of features to be extracted has not size one!" << endl;
        return -1;
    }

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results = feature_extraction_net->Forward();

    const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

    int batch_size = feature_blob->num(); // should be 1
    if (batch_size!=1)
    {
        cout << "Error! Retrieved more than one feature, exiting..." << endl;
        return -1;
    }

    int channels = feature_blob->channels();
    int width = feature_blob->width();
    int height = feature_blob->height();

    if (features==NULL)
    {
        features = new Blob<Dtype>(1, channels, height, width);
    } else
    {
        features->Reshape(1, channels, height, width);
    }

    features->CopyFrom(*feature_blob);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        return msecTotal;
    }
    else
    {
        return 0;
    }
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_multipleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features)
{

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize the labels to zero
    vector<int> labels(images.size(), 0);

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size

    if (memory_data_layer->batch_size()!=new_batch_size)
    {
        if (images.size()%new_batch_size==0)
        {
            memory_data_layer->set_batch_size(new_batch_size);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
        else
        {
            if (images.size()%memory_data_layer->batch_size()==0)
            {
                cout << "WARNING: image number is not multiple of requested batch size,leaving the old one." << endl;
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            } else
            {
                cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
                memory_data_layer->set_batch_size(1);
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            }

        }

    } else
    {
        if (images.size()%memory_data_layer->batch_size()!=0)
        {
            cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
            memory_data_layer->set_batch_size(1);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
    }

    int num_batches = images.size()/new_batch_size;

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied


    for (int i=0; i<images.size(); i++)
    {
        if (images[i].rows != mean_height || images[i].cols != mean_height)
        {
            if (images[i].rows > mean_height || images[i].cols > mean_height)
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
            }
            else
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
            }
        }
    }

    memory_data_layer->AddMatVector(images,labels);

    size_t num_features = blob_names.size();

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results;

    for (int b=0; b<num_batches; ++b)
    {
        results = feature_extraction_net->Forward();

        for (int f = 0; f < num_features; ++f) {

            const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

            int batch_size = feature_blob->num();

            int feat_dim = feature_blob->count() / batch_size; // should be equal to channels*width*height

            if (feat_dim!=feature_blob->channels()) // the feature is not 1D i.e. width!=1 or height!=1
            {
                cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
            }

            for (int i=0; i<batch_size; ++i)
            {
                features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(i), feature_blob->mutable_cpu_data() + feature_blob->offset(i) + feat_dim));
            }

        }

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        float msecPerImage = msecTotal/(float)images.size();

        return msecPerImage;
    }
    else
    {
        return 0;
    }
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extractBatch_singleFeat_1D(vector<cv::Mat> &images, int new_batch_size, vector< vector<Dtype> > &features)
{

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize labels to zero
    vector<int> labels(images.size(), 0);

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size

    if (memory_data_layer->batch_size()!=new_batch_size)
    {
        if (images.size()%new_batch_size==0)
        {
            memory_data_layer->set_batch_size(new_batch_size);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
        else
        {
            if (images.size()%memory_data_layer->batch_size()==0)
            {
                cout << "WARNING: image number is not multiple of requested batch size,leaving the old one." << endl;
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            } else
            {
                cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
                memory_data_layer->set_batch_size(1);
                cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
            }

        }

    } else
    {
        if (images.size()%memory_data_layer->batch_size()!=0)
        {
            cout << "WARNING: image number is not multiple of batch size, setting it to 1 (performance issue)." << endl;
            memory_data_layer->set_batch_size(1);
            cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
        }
    }

    int num_batches = images.size()/new_batch_size;

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied

    for (int i=0; i<images.size(); i++)
    {
        if (images[i].rows != mean_height || images[i].cols != mean_height)
        {
            if (images[i].rows > mean_height || images[i].cols > mean_height)
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
            }
            else
            {
                cv::resize(images[i], images[i], cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
            }
        }
    }

    memory_data_layer->AddMatVector(images,labels);

    size_t num_features = blob_names.size();
    if (num_features!=1)
    {
        cout<< "Error! The list of features to be extracted has not size one!" << endl;
        return -1;
    }

    // Run network and retrieve features!

    std::vector<Blob<Dtype>*> results;

    for (int b=0; b<num_batches; ++b)
    {
        results = feature_extraction_net->Forward();

        const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

        int batch_size = feature_blob->num();

        int feat_dim = feature_blob->count() / batch_size; // should be equal to: channels*width*height
        if (feat_dim!=feature_blob->channels())
        {
            cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
        }

        for (int i=0; i<batch_size; ++i)
        {
            features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(i), feature_blob->mutable_cpu_data() + feature_blob->offset(i) + feat_dim));
        }

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        float msecPerImage = msecTotal/(float)images.size();

        return msecPerImage;
    }
    else
    {
        return 0;
    }
}

template<class Dtype>
float CaffeFeatExtractor<Dtype>::extract_multipleFeat_1D(cv::Mat &image, vector< vector<Dtype> > &features)
{

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    cudaEvent_t start, stop;

    if (timing)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, NULL);
    }

    // Initialize labels to zero
    int label = 0;

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1

    if (memory_data_layer->batch_size()!=1)
    {
        memory_data_layer->set_batch_size(1);
        cout << "BATCH SIZE = " << memory_data_layer->batch_size() << endl;
    }

    // Input preprocessing

    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied

    if (image.rows != mean_height || image.cols != mean_height)
    {
        if (image.rows > mean_height || image.cols > mean_height)
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
        }
    }

    memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

    size_t num_features = blob_names.size();

    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results = feature_extraction_net->Forward();

    for (int f = 0; f < num_features; ++f) {

        const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[f]);

        int batch_size = feature_blob->num(); // should be 1
        if (batch_size!=1)
        {
            cout << "Error! Retrieved more than one feature, exiting..." << endl;
            return -1;
        }

        int feat_dim = feature_blob->count(); // should be equal to: count/batch_size=channels*width*height
        if (feat_dim!=feature_blob->channels())
        {
            cout<< "Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, width, height)" << endl;
        }

        features.push_back(vector <Dtype>(feature_blob->mutable_cpu_data() + feature_blob->offset(0), feature_blob->mutable_cpu_data() + feature_blob->offset(0) + feat_dim));

    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stop, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stop);

        float msecTotal = 0.0f;
        cudaEventElapsedTime(&msecTotal, start, stop);

        return msecTotal;
    }
    else
    {
        return 0;
    }

}

template<class Dtype>
bool CaffeFeatExtractor<Dtype>::extract_singleFeat_1D(cv::Mat &image, vector<Dtype> &features, float (&times)[2])
{

    times[0] = 0.0f;
    times[1] = 0.0f;

    // Check input image
    if (image.empty())
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): empty imMat!" << std::endl;
        return false;
    }

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

    // Prepare Caffe

    // Set the GPU/CPU mode for Caffe (here in order to be thread-safe)
    if (gpu_mode)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(device_id);
    }
    else
    {
        Caffe::set_mode(Caffe::CPU);
    }

    // Initialize labels to zero
    int label = 0;

    // Get pointer to data layer to set the input
    caffe::shared_ptr<MemoryDataLayer<Dtype> > memory_data_layer = boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype> >(feature_extraction_net->layers()[0]);

    // Set batch size to 1
    if (memory_data_layer->batch_size()!=1)
    {
        memory_data_layer->set_batch_size(1);
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): BATCH SIZE = " << memory_data_layer->batch_size() << std::endl;
    }

    // Image preprocessing
    // The image passed to AddMatVector must be same size as the mean image
    // If not, it is resized:
    // if it is downsampled, an anti-aliasing Gaussian Filter is applied
    if (image.rows != mean_height || image.cols != mean_height)
    {
        if (image.rows > mean_height || image.cols > mean_height)
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LANCZOS4);
        }
        else
        {
            cv::resize(image, image, cv::Size(mean_height, mean_width), 0, 0, CV_INTER_LINEAR);
        }
    }

    memory_data_layer->AddMatVector(vector<cv::Mat>(1, image),vector<int>(1,label));

    size_t num_features = blob_names.size();
    if(num_features!=1)
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): Error! The list of features to be extracted has not size one!" << std::endl;
        return false;
    }

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopPrep, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopPrep);

        cudaEventElapsedTime(times, startPrep, stopPrep);
    }


    // Run network and retrieve features!

    // depending on your net's architecture, the blobs will hold accuracy and/or labels, etc
    std::vector<Blob<Dtype>*> results = feature_extraction_net->Forward();

    const caffe::shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net->blob_by_name(blob_names[0]);

    int batch_size = feature_blob->num(); // should be 1
    if (batch_size!=1)
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): Error! Retrieved more than one feature, exiting..." << std::endl;
        return -1;
    }

    int feat_dim = feature_blob->count(); // should be equal to: count/num=channels*width*height
    if (feat_dim!=feature_blob->channels())
    {
        std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): Attention! The feature is not 1D: unrolling according to Caffe's order (i.e. channel, height, width)" << std::endl;
    }

    features.insert(features.end(), feature_blob->mutable_cpu_data() + feature_blob->offset(0), feature_blob->mutable_cpu_data() + feature_blob->offset(0) + feat_dim);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopNet, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopNet);

        cudaEventElapsedTime(times+1, startNet, stopNet);

    }

    return true;

}

#endif /* CAFFEFEATEXTRACTOR_H_ */
