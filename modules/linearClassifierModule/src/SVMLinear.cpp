#include "SVMLinear.h"

SVMLinear::SVMLinear(string className) {

    this->className=className;
    modelLinearSVM=NULL;


}

void SVMLinear::trainModel(std::vector<std::vector<double> > &features, vector<double> &labels, parameter &param,int bias) {


    SVMProblem.bias=bias;
    SVMProblem.l=features.size();
    SVMProblem.n=features[0].size()+bias; //+1 bias term

    SVMProblem.y = Malloc(double,SVMProblem.l);
    SVMProblem.x = Malloc(struct feature_node *,SVMProblem.l);

    
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

        SVMProblem.x[i]=Malloc(struct feature_node,sparsity+bias+1); //bias and -1 index
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

        if(bias)
        {
            SVMProblem.x[i][cnt].index=SVMProblem.n,
            SVMProblem.x[i][cnt].value=1;
            cnt++;
        }
        SVMProblem.x[i][cnt].index=-1;


    }


    modelLinearSVM=train(&SVMProblem, &param);

    
    free(SVMProblem.y);
    for (int i=0; i<SVMProblem.l; i++)
        free(SVMProblem.x[i]);
    free(SVMProblem.x);

    if(param.nr_weight>0)
    {
        free(param.weight);
        free(param.weight_label);
    }

}


void SVMLinear::saveModel(string pathFile)
{
    save_model(pathFile.c_str(), modelLinearSVM);
}

void SVMLinear::loadModel(string pathFile)
{
    modelLinearSVM=load_model(pathFile.c_str());
}


/*solver_type can be one of L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL.

    L2R_LR                L2-regularized logistic regression (primal)
    L2R_L2LOSS_SVC_DUAL   L2-regularized L2-loss support vector classification (dual)
    L2R_L2LOSS_SVC        L2-regularized L2-loss support vector classification (primal)
    L2R_L1LOSS_SVC_DUAL   L2-regularized L1-loss support vector classification (dual)
    MCSVM_CS              multi-class support vector classification by Crammer and Singer
    L1R_L2LOSS_SVC        L1-regularized L2-loss support vector classification
    L1R_LR                L1-regularized logistic regression
    L2R_LR_DUAL           L2-regularized logistic regression (dual)
    L2R_L2LOSS_SVR        L2-regularized L2-loss support vector regression (primal)
    L2R_L2LOSS_SVR_DUAL   L2-regularized L2-loss support vector regression (dual)
    L2R_L1LOSS_SVR_DUAL   L2-regularized L1-loss support vector regression (dual)
    
    C= contraints violation Default 1

    eps is the stopping criterion:

    L2R_LR and L2R_L2LOSS_SVC
		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
		where f is the primal function and pos/neg are # of
		positive/negative data (default 0.01)

	L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS and L2R_LR_DUAL

		Dual maximal violation <= eps; similar to libsvm (default 0.1)
	L1R_L2LOSS_SVC and L1R_LR
		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
		where f is the primal function (default 0.01)


    p is the sensitiveness of loss of support vector regression. 
    */
parameter SVMLinear::initialiseParam(int solverTYPE, double C, double eps, int nClass, int nr_Positive, int nr_Negative)
{
    parameter param;
    param.solver_type=solverTYPE;
    param.C=C;
    param.eps=eps;
    if(nr_Negative==0 || nr_Negative<nr_Positive)
    {
        param.nr_weight=0;
        return param;
    }

    param.nr_weight=(nClass);
    param.weight_label = Malloc(int,nClass);
    param.weight= Malloc(double,nClass);


    param.weight_label[0]=1;
    param.weight[0]=nr_Negative/nr_Positive;


    for (int i=1; i<nClass; i++)
    {
        param.weight_label[i]=-1;
        param.weight[i]=1.0;
    }

    return param;

}

double SVMLinear::predictModel(vector<double> features)
{

    if(modelLinearSVM==NULL)
    {
        fprintf(stdout,"Error, Train Model First \n");
        return 0.0;
    }
    int nr_class=get_nr_class(modelLinearSVM);
    int bias=modelLinearSVM->bias;

    int sparsity=0.0;
    for (int i=0; i<features.size(); i++)
        if(features[i]!=0.0)
            sparsity++;

    feature_node *x=Malloc(struct feature_node,sparsity+bias+1); //bias and -1 index

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
    if(bias)
    {
        x[cnt].index=modelLinearSVM->nr_feature+1,
        x[cnt].value=1;
        cnt++;
    }
    x[cnt].index=-1;

    double val=0;
    predict_values(modelLinearSVM,x,&val);
    free(x);
    return val;

}


vector<vector<double> > SVMLinear::readFeatures(string filePath)
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



