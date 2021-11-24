#include "DictionaryLearning.h"

using namespace std;
using namespace yarp::sig;
using namespace yarp::os;
using namespace yarp::math;

DictionaryLearning::DictionaryLearning(string dictionariesFilePath, string group, string codingM, int K)
{
    this->dictionariesFilePath=dictionariesFilePath;
    this->group=group;
    this->codingType=codingM;
    
    read();

    maxComparisons=350;
    cv::Mat dict;

    dictionaryT=dictionary.transposed();
    convert(dictionaryT,dict);
    kdtree.build(dict);
    if(usePCA)
    {
        feat.resize(dimPCA);
    }
    else
    {
        feat.resize(featuresize);
    }
    Knn=K;
    rankIndices.resize(Knn);

    fprintf(stdout, "featureSize %d, dictionarySize %d, usePCA %s, dimPCA %d, KNN %d, powerNormalization %s, alpha %f, Mapping %s \n",featuresize,dictionarySize,(usePCA)?"true":"false",dimPCA, Knn, (powerNormalization)?"true":"false",alpha,mappingType.c_str());

}

bool DictionaryLearning::read()
{
    Property config; config.fromConfigFile(dictionariesFilePath.c_str());
    Bottle& bgeneral=config.findGroup("GENERAL");
    lambda=bgeneral.find("lambda").asFloat64();
    usePCA=bgeneral.check("usePCA",Value(0)).asInt32();
    dimPCA=bgeneral.check("dimPCA",Value(80)).asInt32();

    powerNormalization=bgeneral.check("powerNormalization",Value(0)).asInt32();
    alpha=bgeneral.check("alpha",Value(3)).asInt32(); // alpha-root
    alpha=1./alpha;


    mappingType=bgeneral.check("mappingType",Value("linear")).asString().c_str(); 

    dictionarySize=bgeneral.find("dictionarySize").asInt32();
    Bottle& dictionaryGroup=config.findGroup(group.c_str());
    featuresize=dictionaryGroup.find("featureSize").asInt32();
    if(!usePCA)
        dictionary.resize(featuresize,dictionarySize);
    else
        dictionary.resize(dimPCA,dictionarySize);


    int fdim=usePCA? dimPCA:featuresize;

    for (int i=0; i<fdim; i++)
    {
        ostringstream oss;
        oss << (i+1);
        string num = "line"+oss.str();
        Bottle* line=dictionaryGroup.find(num.c_str()).asList();
        
        for (int j=0; j<dictionarySize; j++)
            dictionary(i,j)=line->get(j).asFloat64();
    }

    double beta=1e-4;
    Matrix trans=dictionary.transposed();
    Matrix identity=eye(dictionarySize,dictionarySize);
    Matrix diagonal=2*beta*identity;
    A=trans*dictionary+diagonal;



    if(usePCA)
    {
        Bottle& eigengroup=config.findGroup("EIGENVECTORS");
        yarp::sig::Matrix eig(dimPCA,featuresize);
        for (int i=0; i<dimPCA; i++)
        {
            ostringstream oss;
            oss << (i+1);
            string num = "line"+oss.str();
            Bottle* line=eigengroup.find(num.c_str()).asList();
            
            for (int j=0; j<featuresize; j++)
                eig(i,j)=line->get(j).asFloat64();
        }

        cv::Mat eigens;
        convert(eig,eigens);
        PCA.eigenvectors=eigens;

        eigengroup=config.findGroup("EIGENVALUES");
        yarp::sig::Matrix eigv(dimPCA,1);
        for (int i=0; i<dimPCA; i++)
        {
            ostringstream oss;
            oss << (i+1);
            string num = "line"+oss.str();
            Bottle* line=eigengroup.find(num.c_str()).asList();
            
            eigv(i,0)=line->get(0).asFloat64();
        }

        cv::Mat eigenvs;
        convert(eigv,eigenvs);
        PCA.eigenvalues=eigenvs;

        eigengroup=config.findGroup("MEAN");
        yarp::sig::Matrix mean(1,featuresize);
        string num = "line1";
        Bottle* line=eigengroup.find(num.c_str()).asList();
    
        for (int j=0; j<featuresize; j++)
            mean(0,j)=line->get(j).asFloat64();
        

        cv::Mat m;
        convert(mean,m);
        PCA.mean=m;
    }
    return true;
}

void DictionaryLearning::isBestElement(double score, int index, int numberOfBestPoints)
{
    if (rankScores.size()==0)
    {
        rankScores.push_back(score);
        rankIndices.push_back(index);
    }
    else if (rankScores.size()<numberOfBestPoints)
    {
        bool assigned=false;
        std::vector<int>::iterator itind=rankIndices.begin();
        for (std::vector<double>::iterator itsc = rankScores.begin(); itsc!=rankScores.end(); itsc++)
        {
            if (*itsc<score)
            {
                rankScores.insert(itsc,score);
                rankIndices.insert(itind,index);
                assigned=true;
                break;
            }
            itind++;
        }
        if (!assigned)
        {
            rankScores.push_back(score);
            rankIndices.push_back(index);
        }
    }
    else
    {
        if (rankScores[rankScores.size()-1]>score)
            return;
        else if (rankScores[0]<score)
        {
            std::vector<double>::iterator itsc=rankScores.begin();
            std::vector<int>::iterator itind=rankIndices.begin();
            rankScores.insert(itsc,score);
            rankIndices.insert(itind,index);
            rankScores.pop_back();
            rankIndices.pop_back();
        }
        else
        {
            std::vector<int>::iterator itind=rankIndices.begin();
            for (std::vector<double>::iterator itsc = rankScores.begin(); itsc!=rankScores.end(); itsc++)
            {
                if (*itsc<score)
                {
                    rankScores.insert(itsc,score);
                    rankIndices.insert(itind,index);
                    rankScores.pop_back();
                    rankIndices.pop_back();
                    break;
                }
                itind++;
            }
        }
    }
}

void DictionaryLearning::bestCodingEver_ANN(yarp::sig::Vector& feature, yarp::sig::Vector& descriptor)
{
    if(usePCA)
    {
        cv::Mat mat(1,feature.size(),CV_64FC1,(double*) feature.data());
        mat=PCA.project(mat);
        feature.resize(dimPCA,0.0);
        for (int i=0; i<feature.size(); i++)
            feature[i]=mat.at<float>(0,i);
    }

    descriptor.resize(dictionarySize,0.0);
    for (int i=0; i<feature.size(); i++)
    {
         feat[i]=feature[i];
    }
    
    kdtree.findNearest(feat,Knn,maxComparisons,rankIndices);

    for (int i=0; i<Knn; i++)
    {
        int id=rankIndices[i];
        
        yarp::sig::Vector v=dictionaryT.getRow(id);
        double val;
        if(mappingType=="linear")
            val=yarp::math::dot(v,feature);

        if(mappingType=="additive_chisquare")
        {
            val=additive_chisquare(v, feature);
        }

        if(mappingType=="gauss_chisquare")
        {
            val=gauss_chisquare(v, feature);
        }
        descriptor[id]=val;
        
    }


}


void DictionaryLearning::bestCodingEver(const yarp::sig::Vector& feature, yarp::sig::Vector& descriptor)
{
    rankScores.clear();
    rankIndices.clear();
    Vector code=dictionaryT*feature;
    descriptor.resize(code.size(),0.0);

    for (int i=0; i<code.size(); i++)
    {
        isBestElement(code[i], i, Knn);
    }

    for (int i=0; i<Knn; i++)
    {
        int id=rankIndices[i];
        descriptor[id]=rankScores[i];
    }
    
}





double DictionaryLearning::customNorm(const yarp::sig::Vector &A,const yarp::sig::Vector &B)
{
       double norm=0;

       for (int i=0; i<128; i++)
       {
           double d=A[i]-B[i];
           norm+=d*d;
           
       }
       
       
       return norm;

}

void DictionaryLearning::bow(yarp::sig::Vector & feature, yarp::sig::Vector & code)
{

    if(usePCA)
    {
        cv::Mat mat(1,feature.size(),CV_64FC1,(double*) feature.data());
        mat=PCA.project(mat);
        feature.resize(dimPCA,0.0);
        for (int i=0; i<feature.size(); i++)
            feature[i]=mat.at<float>(0,i);
    }
	
    //transform to array of float
     fprintf(stdout, "Starting bow coding \n");
     double time=Time::now();
     code.resize(dictionarySize);
     code=0.0;


	 float minNorm=1e20;
	 int winnerAtom;


	 for(int atom_idx=0; atom_idx<dictionarySize; atom_idx++)
	 {
		 yarp::sig::Vector v=dictionary.getCol(atom_idx);
		 float norm=0.0f;
		 for (int i=0; i<featuresize; i++)
		 {
			 float d=feature[i]-v[i];
			 norm+=d*d;
			 if (norm > minNorm) 
				 break;
	   
		 }
		 if(norm<minNorm)
		 {
			 minNorm=norm;
			 winnerAtom=atom_idx;
		 }
		 

	 }
	 
	 code[winnerAtom]++;    
     time=Time::now()-time;
     fprintf(stdout, "%f \n",time);
     
 


}


void DictionaryLearning::avgPooling(std::vector<yarp::sig::Vector> & features, yarp::sig::Vector & code, vector<SiftGPU::SiftKeypoint> & keypoints, int pLevels, int imgW, int imgH)
{
     //fprintf(stdout, "Starting max pooling \n");
     double time=Time::now();
     vector<Vector> allCodes;
     allCodes.resize(features.size());
     
     for (int i=0; i<features.size(); i++)
     {
        Vector &current=features[i];



        Vector currentCode;
        if(codingType=="SC")
            computeSCCode(current,currentCode);
        if(codingType=="BCE")
            bestCodingEver_ANN(current,currentCode);
        if(codingType=="BOW")
            bow(current,currentCode);			
         allCodes[i]=currentCode;
     }
     
     //fprintf(stdout, "Finished \n");
     
     int sizeF=allCodes[0].size();
     int sizeS=allCodes.size();
     

     vector<int> pBins(pLevels);
     vector<int> pyramid(pLevels);
     
     pBins[0]=1;
     pyramid[0]=1;
     int tBins=1;
     
     for (int i=1; i<pLevels; i++)
     {
         pyramid[i]=i*2;
         pBins[i]=(i*2)*(i*2);
         tBins+=pBins[i];
     }
     

     
     code.resize(sizeF*tBins);
     int binId=0;
     //cvZero(debugImg);
     for (int iter=0; iter<pLevels; iter++)
     {
              //fprintf(stdout, "Iter: %d \n", iter);
              int nBins=pyramid[iter];
              int wUnit= imgW/pyramid[iter];
              int hUnit= imgH/pyramid[iter];
              
              //fprintf(stdout, "tBins: %d nBins: %d  imgW: %d imgH: %d \n", tBins, nBins,wUnit,hUnit);
              
              for (int iterBinW=0; iterBinW<nBins; iterBinW++)
              {
                   for (int iterBinH=0; iterBinH<nBins; iterBinH++)
                   {
                       int currBinW=wUnit*(iterBinW+1);
                       int currBinH=hUnit*(iterBinH+1);
                       int prevBinW=wUnit*iterBinW;
                       int prevBinH=hUnit*iterBinH;
                       
                       double normalizer=0.0;
                       int start=binId;
                       for (int i=0; i<sizeF; i++)
                       {
                           double value=0.0;
						   int total=0;
                           for (int k=0; k<sizeS; k++)
                           {
                               
                               if(keypoints[k].x< prevBinW || keypoints[k].x>= currBinW || keypoints[k].y< prevBinH || keypoints[k].y>= currBinH)
                                   continue;
                                   
                                   /*if(iter==2 && iterBinW==3 && iterBinH==3)
                                   {
                                       int x = cvRound(keypoints[k].x);
                                       int y = cvRound(keypoints[k].y);
                                       cvCircle(debugImg,cvPoint(x,y),2,cvScalar(255),2);
                                       //fprintf(stdout,"Feature X %f: Feature Y: %f \n", keypoints[k].x,keypoints[k].y);
                                    }*/
                               value=value+allCodes[k][i];
							   total++;
                           }
						        if(total>0)
								{
									code[binId]=value/total;
									normalizer=normalizer+((value/total)*(value/total));
								}
                                binId++;
                        }

                        // Power Normalization
                        if(powerNormalization)
                        {
                            if(normalizer>0)
                                normalizer=sqrt(normalizer);
                            {
                                for (int i=start; i<binId; i++)
                                {
                                    code[i]=code[i]/normalizer;
                                    int sign = code[i]>0? 1:-1;
                                    code[i]=sign*pow(abs(code[i]),alpha);
                                }
                            }
                        }
                   
                   }
              }
     
     }
     
     if(!powerNormalization)
     {
         double norm= yarp::math::norm(code);
         code=code/norm; 
     }

     time=Time::now()-time;
     //fprintf(stdout, "%f \n",time);
}


void DictionaryLearning::maxPooling(std::vector<yarp::sig::Vector> & features, yarp::sig::Vector & code, vector<SiftGPU::SiftKeypoint> & keypoints, int pLevels, int imgW, int imgH)
{
     //fprintf(stdout, "Starting max pooling \n");
     double time=Time::now();
     vector<Vector> allCodes;
     allCodes.resize(features.size());
     
     for (int i=0; i<features.size(); i++)
     {
        Vector &current=features[i];



        Vector currentCode;
        if(codingType=="SC")
            computeSCCode(current,currentCode);
        if(codingType=="BCE")
            bestCodingEver_ANN(current,currentCode);
        if(codingType=="BOW")
            bow(current,currentCode);			
         allCodes[i]=currentCode;
     }
     
     //fprintf(stdout, "Finished \n");
     
     int sizeF=allCodes[0].size();
     int sizeS=allCodes.size();
     

     vector<int> pBins(pLevels);
     vector<int> pyramid(pLevels);
     
     pBins[0]=1;
     pyramid[0]=1;
     int tBins=1;
     
     for (int i=1; i<pLevels; i++)
     {
         pyramid[i]=i*2;
         pBins[i]=(i*2)*(i*2);
         tBins+=pBins[i];
     }
     

     
     code.resize(sizeF*tBins);
     int binId=0;
     //cvZero(debugImg);
     for (int iter=0; iter<pLevels; iter++)
     {
              //fprintf(stdout, "Iter: %d \n", iter);
              int nBins=pyramid[iter];
              int wUnit= imgW/pyramid[iter];
              int hUnit= imgH/pyramid[iter];
              
              //fprintf(stdout, "tBins: %d nBins: %d  imgW: %d imgH: %d \n", tBins, nBins,wUnit,hUnit);
              
              for (int iterBinW=0; iterBinW<nBins; iterBinW++)
              {
                   for (int iterBinH=0; iterBinH<nBins; iterBinH++)
                   {
                       int currBinW=wUnit*(iterBinW+1);
                       int currBinH=hUnit*(iterBinH+1);
                       int prevBinW=wUnit*iterBinW;
                       int prevBinH=hUnit*iterBinH;
                       
                       double normalizer=0.0;
                       int start=binId;
                       for (int i=0; i<sizeF; i++)
                       {
                           double maxVal=-1000.0;
                           for (int k=0; k<sizeS; k++)
                           {
                               
                               if(keypoints[k].x< prevBinW || keypoints[k].x>= currBinW || keypoints[k].y< prevBinH || keypoints[k].y>= currBinH)
                                   continue;
                                   
                                   /*if(iter==2 && iterBinW==3 && iterBinH==3)
                                   {
                                       int x = cvRound(keypoints[k].x);
                                       int y = cvRound(keypoints[k].y);
                                       cvCircle(debugImg,cvPoint(x,y),2,cvScalar(255),2);
                                       //fprintf(stdout,"Feature X %f: Feature Y: %f \n", keypoints[k].x,keypoints[k].y);
                                    }*/
                               double val=allCodes[k][i];
                               if(val>maxVal)
                                   maxVal=val;
                           }
                                code[binId]=maxVal;
                                normalizer=normalizer+maxVal*maxVal;
                                binId++;
                        }

                        // Power Normalization
                        if(powerNormalization)
                        {
                            if(normalizer>0)
                                normalizer=sqrt(normalizer);
                            {
                                for (int i=start; i<binId; i++)
                                {
                                    code[i]=code[i]/normalizer;
                                    int sign = code[i]>0? 1:-1;
                                    code[i]=sign*pow(abs(code[i]),alpha);
                                }
                            }
                        }
                   
                   }
              }
     
     }
     
     if(!powerNormalization)
     {
         double norm= yarp::math::norm(code);
         code=code/norm; 
     }

     time=Time::now()-time;
     //fprintf(stdout, "%f \n",time);
}


void DictionaryLearning::computeSCCode(yarp::sig::Vector& feature, yarp::sig::Vector& descriptor)
{
    if(usePCA)
    {
        cv::Mat mat(1,feature.size(),CV_64FC1,(double*) feature.data());
        mat=PCA.project(mat);
        feature.resize(dimPCA,0.0);
        for (int i=0; i<feature.size(); i++)
            feature[i]=mat.at<float>(0,i);
    }
	
    double eps=1e-7;
    Vector b=(-1)*feature*dictionary;

    Vector grad=b;
    descriptor.resize(dictionarySize,0.0);
    double maxValue; int index=-1;
    max(grad,maxValue,index);

    while(true)
    {
        if (grad[index]>lambda+eps)
            descriptor[index]=(lambda-grad[index])/A(index,index);
        else if (grad[index]<-lambda-eps)
            descriptor[index]=(-lambda-grad[index])/A(index,index);
        else
        {
            double sum=0.0;
            for (int j=0; j<descriptor.size(); j++)
                sum+=descriptor[j];
            if (sum==0.0)
                break;
        }

        while(true)
        {
            Matrix Aa;
            Vector ba;
            Vector xa;
            Vector a;
            Vector signes;

            for (int i=0; i<descriptor.size(); i++)
            {
                if (descriptor[i]!=0.0)
                {
                    a.push_back(i);
                    ba.push_back(b[i]);
                    xa.push_back(descriptor[i]);
                    if (descriptor[i]>0)
                        signes.push_back(1);
                    else
                        signes.push_back(-1);
                }
            }

            subMatrix(A,a,Aa);

            Vector vect=(-lambda*signes)-ba;
            Vector xnew=luinv(Aa)*vect;

            Vector vectidx;
            Vector bidx;
            Vector xnewidx;

            double sum=0.0;
            for (int i=0; i<xnew.size(); i++)
            {
                if (xnew[i]!=0.0)
                {
                    vectidx.push_back(vect[i]);
                    bidx.push_back(ba[i]);
                    xnewidx.push_back(xnew[i]);
                    sum+=abs(xnew[i]);
                }
            }

            double onew=dot((vectidx/2+bidx),xnewidx)+lambda*sum;

            Vector s;
            bool changeSign=false;
            for (int i=0; i<xa.size(); i++)
            {
                if (xa[i]*xnew[i]<=0)
                {
                    changeSign=true;
                    s.push_back(i);
                }
            }

            if (!changeSign)
            {
                for (int i=0; i<a.size(); i++)
                    descriptor[a[i]]=xnew[i];
                break;
            }

            Vector xmin=xnew;
            double omin=onew;

            Vector d=xnew-xa;
            Vector t=d/xa;

            for (int i=0; i<s.size(); i++)
            {
                Vector x_s=xa-d/t[s[i]];
                x_s[s[i]]=0.0;

                Vector x_sidx;
                Vector baidx;
                Vector idx;
                sum=0.0;
                for (int j=0; j<x_s.size(); j++)
                {
                    if (x_s[j]!=0)
                    {
                        idx.push_back(j);
                        sum+=abs(x_s[j]);
                        x_sidx.push_back(x_s[j]);
                        baidx.push_back(ba[j]);
                    }
                }
                Matrix Atmp2;
                subMatrix(Aa,idx,Atmp2);

                Vector otmp=Atmp2*(x_sidx/2)+baidx;
                double o_s=dot(otmp,x_sidx)+lambda*sum;

                if (o_s<omin)
                {
                    xmin=x_s;
                    omin=o_s;
                }
            }

            for (int i=0; i<a.size(); i++)
                descriptor[a[i]]=xmin[i];
        }
        //grad=A*descriptor+b;

        // We can perform this operation since the matrix A is symmetric! 10 times faster
        grad=descriptor;
        grad*=A;
        grad+=b;

        Vector tmp;

        for (int i=0; i<descriptor.size(); i++)
        {
            if (descriptor[i]==0.0)
                tmp.push_back(grad[i]);
            else
                tmp.push_back(0);
        }
        max(tmp,maxValue,index);

        if (maxValue<=lambda+eps)
            break;
    }
}

void DictionaryLearning::subMatrix(const Matrix& A, const Vector& indexes, Matrix& Atmp)
{
    int indSize=indexes.size();
    Atmp.resize(indSize,indSize);
    int m=0;
    int n=0;
    for (int i=0; i<indSize; i++)
    {
        for (int j=0; j<indSize; j++)
        {
            Atmp(m,n)=A(indexes[i],indexes[j]);
            n++;
        }
        m++;
        n=0;
    }
}

void DictionaryLearning::max(const Vector& x, double& maxValue, int& index)
{
    maxValue=-1000;
    for (int i=0; i<x.size(); i++)
    {
        if(abs((double)x[i])>maxValue)
        {
            maxValue=abs(x[i]);
            index=i;
        }
    }
}
vector<vector<double> > DictionaryLearning::readFeatures(std::string filePath)
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
void DictionaryLearning::learnDictionary(string featureFile, int dictionarySize, bool usePCA, int dim)
{
    dimPCA=dim;
    this->usePCA=usePCA;

    vector<vector<double> > Features=readFeatures(featureFile);
    if(Features.size()==0)
        return;

    this->featuresize=Features[0].size();

    cv::Mat samples(Features.size(), featuresize,CV_32F);
    

    for (int i=0; i<Features.size(); i++)
    {
        for (int j=0; j<Features[i].size(); j++)
        {
            samples.at<float>(i,j)=(float)Features[i][j];
        }
    }

    if(usePCA)
    {
        fprintf(stdout, "Using PCA with dim: %d \n", dimPCA);
        PCA(samples,cv::Mat(), CV_PCA_DATA_AS_ROW,dimPCA);
        samples=PCA.project(samples);
    }

    int feaSize=usePCA? dimPCA:featuresize;
    cv::Mat labels;
    cv::Mat centroids(dictionarySize,feaSize,CV_32F);
    fprintf(stdout, "Learning Dictionary with %d atoms. \n", dictionarySize);
    
    //cv::kmeans(samples,dictionarySize,labels, cv::TermCriteria(),3,cv::KMEANS_PP_CENTERS  , &centroids);
    cv::kmeans(samples,dictionarySize,labels, cv::TermCriteria(),3,cv::KMEANS_PP_CENTERS  , centroids);

    dictionary.resize(feaSize,dictionarySize);

    if(!usePCA)
    {
        for(int j=0; j<dictionary.cols(); j++)
        {
            for (int i=0; i<dictionary.rows(); i++)
            {
                dictionary(i,j)=centroids.at<float>(j,i);
            }

            Vector col =dictionary.getCol(j);
            double normVal=norm(col);

            if(normVal==0)
                continue;

            for (int i=0; i<dictionary.rows(); i++)
            {
                dictionary(i,j)=dictionary(i,j)/normVal;
            }
            Vector col2 =dictionary.getCol(j);
            normVal=norm(col2);
        }
    }
    dictionaryT=dictionary.transposed();
    double beta=1e-4;
    Matrix trans=dictionary.transposed();
    Matrix identity=eye(dictionarySize,dictionarySize);
    Matrix diagonal=2*beta*identity;
    A=trans*dictionary+diagonal;

    cv::Mat dict;
    convert(dictionaryT,dict);
    kdtree.build(dict);

    this->dictionarySize=dictionarySize;

    if(usePCA)
    {
        feat.resize(dimPCA);
    }
    else
    {
        feat.resize(featuresize);
    }
    
}
bool DictionaryLearning::saveDictionary(string filePath)
{
    ofstream dictfile;
    dictfile.open(filePath.c_str(),ios::trunc);

    dictfile << "[GENERAL] \n";
    dictfile << "lambda " << lambda << "\n";
    int use= usePCA? 1:0;
    dictfile << "usePCA " << use << "\n";
    dictfile << "dimPCA " << dimPCA << "\n";

    dictfile << "dictionarySize " << dictionaryT.rows() << "\n";

    dictfile << "[DICTIONARY] \n";
    dictfile << "featureSize " << featuresize << "\n";


    for (int i=0; i<dictionary.rows(); i++)
    {
        dictfile << "line" << i+1 << " ( ";
        for (int j=0; j<dictionary.cols(); j++)
        {
            dictfile << dictionary(i,j) << " ";
        }
        dictfile << " ) \n";

    }

    if(usePCA)
    {
        dictfile << "[EIGENVECTORS] \n";
        for (int i=0; i<PCA.eigenvectors.rows; i++)
        {
            dictfile << "line" << i+1 << " ( ";
            for (int j=0; j<PCA.eigenvectors.cols; j++)
            {
                dictfile << PCA.eigenvectors.at<float>(i,j) << " ";
            }
            dictfile << " ) \n";

        }

        dictfile << "[EIGENVALUES] \n";
        for (int i=0; i<PCA.eigenvalues.rows; i++)
        {
            dictfile << "line" << i+1 << " ( ";
            for (int j=0; j<PCA.eigenvalues.cols; j++)
            {
                dictfile << PCA.eigenvalues.at<float>(i,j) << " ";
            }
            dictfile << " ) \n";

        }

        dictfile << "[MEAN] \n";
        for (int i=0; i<PCA.mean.rows; i++)
        {
            dictfile << "line" << i+1 << " ( ";
            for (int j=0; j<PCA.mean.cols; j++)
            {
                dictfile << PCA.mean.at<float>(i,j) << " ";
            }
            dictfile << " ) \n";

        }
    }


    dictfile.close();

    return true;
}

void DictionaryLearning::printMatrixYarp(const yarp::sig::Matrix &A) 
{
    cout << endl;
    for (int i=0; i<A.rows(); i++) 
        for (int j=0; j<A.cols(); j++) 
            cout<<A(i,j)<<" ";
        cout<<endl;
    cout << endl;
}


DictionaryLearning::~DictionaryLearning()
{
    //write();
}

void DictionaryLearning::convert(yarp::sig::Matrix& matrix, cv::Mat& mat) {
    mat=cv::Mat(matrix.rows(),matrix.cols(),CV_32FC1);
    for(int i=0; i<matrix.rows(); i++)
        for(int j=0; j<matrix.cols(); j++)
            mat.at<float>(i,j)=matrix(i,j);
}

void DictionaryLearning::convert(cv::Mat& mat, yarp::sig::Matrix& matrix) {
    matrix.resize(mat.rows,mat.cols);
    for(int i=0; i<mat.rows; i++)
        for(int j=0; j<mat.cols; j++)
            matrix(i,j)=mat.at<double>(i,j);
}

double DictionaryLearning::additive_chisquare(const yarp::sig::Vector &x, const yarp::sig::Vector &y){
    double val=0;
    for (int i=0; i<x.size(); i++)
    {
        double res=x[i]*y[i];
        double sumAbs=(abs(x[i])+abs(y[i]));
        if(sumAbs==0)
            continue;
        res=res/sumAbs;
        val=val+res;
    }
    return val;

}

double DictionaryLearning::gauss_chisquare(const yarp::sig::Vector &x, const yarp::sig::Vector &y){
    double val=0;
    for (int i=0; i<x.size(); i++)
    {
        double res=x[i]-y[i];
        int sign=(x[i]*y[i])>0? 1:-1;
        res=res*res;
        double sumAbs=(abs(x[i])+abs(y[i]));
        if(sumAbs==0)
            continue;
        res=res/sumAbs;
        val=val+(sign*exp(-(res*res)));
    }
    return val;

}
