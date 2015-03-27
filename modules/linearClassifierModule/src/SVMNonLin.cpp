#include "SVMNonLin.h"

SVMNonLin::SVMNonLin(string className) {

    this->className=className;
    modelSVM=NULL;


}
SVMNonLin::~SVMNonLin()
{
}

void SVMNonLin::trainModel(std::vector<std::vector<double> > &features, vector<double> &labels, svm_parameter &param) {


    SVMProblem.l=features.size();

    SVMProblem.y = Malloc(double,SVMProblem.l);
    SVMProblem.x = Malloc(struct svm_node *,SVMProblem.l);

    
    for (int i=0; i<features.size(); i++)
    {
        SVMProblem.y[i]=labels[i];
        vector<double> x=features[i];
        int sparsity=0;
        for (int j=0; j<x.size(); j++)
        {
           double value=x[j];
           if(value!=0)
                sparsity++;
        }

        SVMProblem.x[i]=Malloc(struct svm_node,sparsity+1); //-1 index
        int cnt=0;
        for (int j=0; j<x.size(); j++)
        {
           double value=x[j];
           if(value==0)
                continue;
           SVMProblem.x[i][cnt].index=j+1;
           SVMProblem.x[i][cnt].value=value;
           cnt++;
        }


        SVMProblem.x[i][cnt].index=-1;


    }


    modelSVM=svm_train(&SVMProblem, &param);

    
    /*free(SVMProblem.y);
    for (int i=0; i<SVMProblem.l; i++)
        free(SVMProblem.x[i]);
    free(SVMProblem.x);*/

}

void SVMNonLin::freeModel()
{
    svm_free_and_destroy_model(&modelSVM);
}

void SVMNonLin::saveModel(string pathFile)
{
    svm_save_model(pathFile.c_str(), modelSVM);
}

void SVMNonLin::loadModel(string pathFile)
{
    modelSVM=svm_load_model(pathFile.c_str());
}



svm_parameter SVMNonLin::initialiseParam(int solverTYPE, double C, double eps, int kernelType, double gamma)
{
    svm_parameter param;
    param.svm_type=solverTYPE;
    param.C=C;
    param.eps=eps;
    param.nr_weight=0;
    param.kernel_type=kernelType;
    param.gamma=gamma;
    
    return param;

}

double SVMNonLin::predictModel(vector<double> features)
{

    if(modelSVM==NULL)
    {
        fprintf(stdout,"Error, Train Model First \n");
        return 0.0;
    }
    int nr_class=svm_get_nr_class(modelSVM);
  
    int sparsity=0.0;
    for (int i=0; i<features.size(); i++)
        if(features[i]!=0.0)
            sparsity++;

    svm_node *x=Malloc(struct svm_node,sparsity+1); //bias and -1 index

    int cnt=0;
    for (int i=0; i<features.size(); i++)
    {
        if(features[i]!=0.0)
        {
            x[cnt].index=i+1;
            x[cnt].value=features[i];
            cnt++;
        }
    }
    x[cnt].index=-1;

    double val=0;
    svm_predict_values(modelSVM,x,&val);
    return val;

}


vector<vector<double> > SVMNonLin::readFeatures(string filePath)
{
    vector<vector<double> > featuresMat;

    string line;
    ifstream infile;
    infile.open (filePath.c_str());
    while(!infile.eof() && infile.is_open()) // To get you all the lines.
        {
            vector<double> f;
            getline(infile,line); // Saves the line in STRING.

            char * val= strtok((char*) line.c_str()," ");

            while(val!=NULL)
            {
            
                double value=atof(val);
                f.push_back(value);
                val=strtok(NULL," ");
            }
            if(f.size()>0)
                featuresMat.push_back(f);
        }
    infile.close();


    return featuresMat;
}



