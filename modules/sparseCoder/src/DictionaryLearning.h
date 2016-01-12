#include <deque>
#include <cstdio>
#include <sstream>
#include <vector>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <yarp/sig/Vector.h>
#include <yarp/os/Bottle.h>
#include <yarp/os/Property.h>
#include <yarp/math/Math.h>
#include <yarp/os/Time.h>

#include "SiftGPU_Extractor.h"


class DictionaryLearning
{
    std::string dictionariesFilePath;
    std::string codingType;
    yarp::sig::Matrix dictionary;
    yarp::sig::Matrix dictionaryT;
    yarp::sig::Matrix A;
    std::string group;

    int Knn;
    double lambda;

    int dictionarySize;
    int featuresize;
    std::vector<double> rankScores;
    std::vector<int> rankIndices;


    cv::KDTree kdtree;
    int maxComparisons;
    
    vector<float> feat;


    std::string mappingType;

    bool powerNormalization;
    double alpha;

    bool usePCA;
    cv::PCA PCA;
    int dimPCA;


    void subMatrix(const yarp::sig::Matrix& A, const yarp::sig::Vector& indexes, yarp::sig::Matrix& Atmp);
    void max(const yarp::sig::Vector& x, double& maxVal, int& index);
    bool read();
    void printMatrixYarp(const yarp::sig::Matrix& A);
    double customNorm(const yarp::sig::Vector &A,const yarp::sig::Vector &B);
    void isBestElement(double score, int index, int numberOfBestPoints);
    std::vector<std::vector<double> > readFeatures(std::string filePath);
    void convert(yarp::sig::Matrix& matrix, cv::Mat& mat);
    void convert(cv::Mat& mat,yarp::sig::Matrix& matrix);

    double additive_chisquare(const yarp::sig::Vector &x, const yarp::sig::Vector &y);
    double gauss_chisquare(const yarp::sig::Vector &x, const yarp::sig::Vector &y);


public:

    DictionaryLearning(std::string dictionariesFilePath, std::string group, std::string codingM="SC", int K=5);
    void computeSCCode(yarp::sig::Vector& feature, yarp::sig::Vector& descriptor);
    void bestCodingEver(const yarp::sig::Vector& feature, yarp::sig::Vector& descriptor);
    void bestCodingEver_ANN(yarp::sig::Vector& feature, yarp::sig::Vector& descriptor);

    void learnDictionary(std::string featureFile, int dictionarySize=512,bool usePCA=false, int dim=80);
    bool saveDictionary(std::string filePath);
    void maxPooling(std::vector<yarp::sig::Vector> & features, yarp::sig::Vector & code, vector<SiftGPU::SiftKeypoint> & keypoints, int pLevels=1, int imgW=320, int imgH=240);

    void avgPooling(std::vector<yarp::sig::Vector> & features, yarp::sig::Vector & code, vector<SiftGPU::SiftKeypoint> & keypoints, int pLevels=1, int imgW=320, int imgH=240);	
    void bow(yarp::sig::Vector & feature, yarp::sig::Vector & code);
    ~DictionaryLearning();
};

