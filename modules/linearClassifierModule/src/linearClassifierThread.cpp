#include "linearClassifierThread.h"



linearClassifierThread::linearClassifierThread(yarp::os::ResourceFinder &rf, Port* commPort)
{
    this->commandPort=commPort;
    currentState=STATE_DONOTHING;

    this->currPath = rf.getHomeContextPath().c_str();
    
    string moduleName = rf.check("name",Value("linearClassifier"), "module name (string)").asString().c_str();
    this->inputFeatures = "/";
    this->inputFeatures += moduleName;
    this->inputFeatures += rf.check("InputPortFeatures",Value("/features:i"),"Input image port (string)").asString().c_str();

    this->outputPortName = "/";
    this->outputPortName += moduleName;
    this->outputPortName += rf.check("OutputPortClassification",Value("/classification:o"),"Input image port (string)").asString().c_str();

    this->outputScorePortName = "/";
    this->outputScorePortName += moduleName;
    this->outputScorePortName += rf.check("OutputPortScores",Value("/scores:o"),"Input image port (string)").asString().c_str();

    this->bufferSize = rf.check("BufferSize",Value(15),"Buffer Size").asInt32();
    this->CSVM = rf.check("CSVM",Value(1.0),"CSVM").asFloat64();
    this->useWeightedSVM= rf.check("WeightedSVM",Value(0),"WeightedSVM").asInt32();

    printf("WeightedSVM: %d \n",useWeightedSVM);

    string dbfolder = rf.check("databaseFolder",Value("database"), "module name (string)").asString().c_str();
    dbfolder="/"+dbfolder;
    this->currPath=this->currPath+dbfolder;
}

bool linearClassifierThread::getClassList(Bottle &b)
{
    lock_guard<mutex> lg(mtx);
    for(int i=0; i<knownObjects.size(); i++)
        b.addString(knownObjects[i].first.c_str());
    return true;
}

bool linearClassifierThread::changeName(const string &old_name, const string &new_name)
{
    lock_guard<mutex> lg(mtx);

    int j=-1; int k=-1;    
    for (size_t i=0; i<knownObjects.size(); i++)
    {
        if (knownObjects[i].first==old_name)
            j=(int)i;
        else if (knownObjects[i].first==new_name)
            k=(int)i;
    }

    if ((j<0) || (k>=0))
        return false;

    knownObjects[j].first=new_name;

    string old_path=currPath+"/"+old_name;
    string new_path=currPath+"/"+new_name;
    yarp::os::rename(old_path.c_str(),new_path.c_str());

    return true;
}

void linearClassifierThread::checkKnownObjects()
{
    knownObjects.clear();
    linearClassifiers.clear();

    if(yarp::os::stat(currPath.c_str()))
    {
        createFullPath(currPath.c_str());
        return;
    }

    vector<string> files;
    getdir(currPath,files);

    for (int i=0; i<files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare("..") || !files[i].compare("svmmodel"))
            continue;
        string objPath=currPath+"/"+files[i];
        
        string svmPath=objPath+"/svmmodel";
        if(!yarp::os::stat(svmPath.c_str()))
        {
            SVMLinear model(files[i]);
            //model.loadModel(svmPath.c_str());
            //linearClassifiers.push_back(model);
        }

        vector<string> featuresFile;
        vector<string> tmpFiles;
        getdir(objPath,featuresFile);
        
        for (int j=0; j< featuresFile.size(); j++)
        {
            if(!featuresFile[j].compare(".") || !featuresFile[j].compare(".."))
                continue;

            string tmp=objPath+"/"+featuresFile[j];
            tmpFiles.push_back(tmp);
        }

        pair<string, vector<string> > obj(files[i],tmpFiles);
        knownObjects.push_back(obj);
    }
}

bool linearClassifierThread::threadInit() 
{
     if (!featuresPort.open(inputFeatures.c_str())) {
        cout  << ": unable to open port " << inputFeatures << endl;
        return false; 
    }

    if (!outputPort.open(outputPortName.c_str())) {
        cout  << ": unable to open port " << outputPortName << endl;
        return false; 
    }

    if (!scorePort.open(outputScorePortName.c_str())) {
        cout  << ": unable to open port " << outputScorePortName << endl;
        return false; 
    }

    trainClassifiersHelper();
    return true;
}

void linearClassifierThread::run()
{
    int current=0;
    while (!isStopping())
    {
        Bottle *p=featuresPort.read();
        if (p==NULL) {
            continue;
        }

        lock_guard<mutex> lg(mtx);

        vector<double> feature;
        feature.resize(p->size());

        for (int i=0; i<p->size(); i++)
            feature[i]=p->get(i).asFloat64();
    
        if(currentState==STATE_DONOTHING)
            continue;

        if(currentState==STATE_SAVING)
        {
            for (int i=0; i<feature.size(); i++)
                objFeatures << feature[i] << " ";
            objFeatures << endl;
        }

        if(currentState==STATE_RECOGNIZING)
        {
            if(linearClassifiers.size()==0)
                continue;

            //cout << "ISTANT SCORES: ";
            double maxVal=-1000;
            double minValue=1000;
            double idWin=-1;
            for(int i =0; i<linearClassifiers.size(); i++)
            {
                double value=linearClassifiers[i].predictModel(feature);
                if(value>maxVal)
                {
                    maxVal=value;
                    idWin=i;
                }
                if(value<minValue)
                    minValue=value;
                bufferScores[current%bufferSize][i]=(value);
                countBuffer[current%bufferSize][i]=0;
                //cout << knownObjects[i].first << " " << value << " ";
            }
            countBuffer[current%bufferSize][idWin]=1;

            vector<double> avgScores(linearClassifiers.size(),0.0);
            vector<double> bufferVotes(linearClassifiers.size(),0.0);            

            for(int i =0; i<bufferSize; i++)
            {
                for(int k =0; k<linearClassifiers.size(); k++)
                {
                    avgScores[k]=avgScores[k]+bufferScores[i][k];
                    bufferVotes[k]=bufferVotes[k]+countBuffer[i][k];
                }
            }

            double maxValue=-100;
            double maxVote=0;
            int indexClass=-1;
            int indexMaxVote=-1;
            
            for(int i =0; i<linearClassifiers.size(); i++)
            {
                avgScores[i]=avgScores[i]/bufferSize;
                if(avgScores[i]>maxValue)
                {
                    maxValue=avgScores[i];
                    indexClass=i;
                }

                if(bufferVotes[i]>maxVote)
                {
                    maxVote=bufferVotes[i];
                    indexMaxVote=i;
                }
            }

            string winnerClass=knownObjects[indexClass].first;
            string winnerVote=knownObjects[indexMaxVote].first;
            
            if(bufferVotes[indexMaxVote]/bufferSize<0.75)
                winnerClass="?";

            if(outputPort.getOutputCount()>0)
            {
                Bottle &b=outputPort.prepare();
                b.clear();
                b.addString(winnerClass.c_str());
                outputPort.write();
            }

            if(scorePort.getOutputCount()>0)
            {
                Bottle allScores;
                for(int i =0; i<linearClassifiers.size(); i++)
                {
                    Bottle &b=allScores.addList();
                    b.addString(knownObjects[i].first.c_str());
                    b.addFloat64(bufferScores[current%bufferSize][i]);
                    
                }
                scorePort.write(allScores);
            }

            current++;
        }
    }
}

void linearClassifierThread::threadRelease() 
{
    if(linearClassifiers.size()>0)
        for (int i=0; i<linearClassifiers.size(); i++)
            linearClassifiers[i].freeModel();

    this->commandPort->close();
    this->featuresPort.close();
    this->outputPort.close();
    this->scorePort.close();
}

void linearClassifierThread::onStop()
{
    this->commandPort->interrupt();
    this->featuresPort.interrupt();
    this->outputPort.interrupt();
    this->scorePort.interrupt();
}

void linearClassifierThread::prepareObjPath(const string &objName)
{    
    lock_guard<mutex> lg(mtx);
    stopAllHelper();

    pathObj=currPath+"/"+objName;
    if(yarp::os::stat(pathObj.c_str()))
    {
        createFullPath(pathObj.c_str());
        pathObj=pathObj+"/1.txt";
    }
    else
    {
        char tmpPath[255];
        bool proceed=true;
       
        for (int i=1; proceed; i++)
        {
           sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
           proceed=!yarp::os::stat(tmpPath);
           sprintf(tmpPath,"%s/%d.txt",pathObj.c_str(),i);
        }

        pathObj=tmpPath;
    }

    objFeatures.open(pathObj.c_str(),fstream::out | fstream::out);
    currentState=STATE_SAVING;    
}

void linearClassifierThread::createFullPath(const char * path)
{
    if (yarp::os::stat(path))
    {
        string strPath=string(path);
        size_t found=strPath.find_last_of("/");
    
        while (strPath[found]=='/')
            found--;

        createFullPath(strPath.substr(0,found+1).c_str());
        yarp::os::mkdir(strPath.c_str());
    }
}

void linearClassifierThread::stopAllHelper()
{
    currentState=STATE_DONOTHING;
    if(objFeatures.is_open())
        objFeatures.close();
}

void linearClassifierThread::stopAll()
{
    lock_guard<mutex> lg(mtx);
    stopAllHelper();
}

int linearClassifierThread::getdir(const string &dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error opening " << dir << endl;
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

bool linearClassifierThread::loadFeatures()
{
    checkKnownObjects();
    if(this->knownObjects.size()==0)
        return false;

    Features.clear();
    Features.resize(knownObjects.size());
    datasetSizes.clear();
    SVMLinear svmmodel(knownObjects[0].first);

    for (int i=0; i<knownObjects.size(); i++)
    {
        vector<string> obj=knownObjects[i].second;
        int cnt=0;
        for (int k=0; k< obj.size(); k++)
        {
            vector<vector<double> > tmpF=svmmodel.readFeatures(obj[k]);
            cnt=cnt+tmpF.size();
            for (int t =0; t<tmpF.size(); t++)
                Features[i].push_back(tmpF[t]);
    
        }

        if (cnt>0)
            this->datasetSizes.push_back(cnt);
    }
    
    return true;
}

bool linearClassifierThread::trainClassifiersHelper()
{
    stopAllHelper();

    if(linearClassifiers.size()>0)
        for (int i=0; i<linearClassifiers.size(); i++)
            linearClassifiers[i].freeModel();

    cout << "load features" << endl;
    loadFeatures();
    if(this->datasetSizes.size()==0)
        return false;

    cout << "features loaded" << endl;

    linearClassifiers.clear();

    int nClass=2; // One vs All

    for (int i=0; i<knownObjects.size(); i++)
    {
        int nPositive=0;
        int nNegative=0;
        string name=knownObjects[i].first;
        SVMLinear svmmodel(name);
        vector<vector<double> > orderedF;
        vector<double> orderedLabels;
        for (int k=0; k<knownObjects.size(); k++)
        {
            for(int j=0; j<Features[k].size(); j++)
                if(knownObjects[i].first==knownObjects[k].first)
                {
                    orderedF.push_back(Features[k][j]);
                    orderedLabels.push_back(1.0);
                    nPositive++;
                }
        }

        for (int k=0; k<knownObjects.size(); k++)
        {
            for(int j=0; j<Features[k].size(); j++)
                if(knownObjects[i].first!=knownObjects[k].first)
                {
                    orderedF.push_back(Features[k][j]);
                    orderedLabels.push_back(-1.0);
                    nNegative++;
                }
        }

        printf("nClass: %d nPositive: %d nNegative: %d\n",nClass,nPositive,nNegative);
        parameter param;
        if(useWeightedSVM)
            param=svmmodel.initialiseParam(L2R_L2LOSS_SVC,CSVM,0.0,nClass,nPositive,nNegative);
        else
            param=svmmodel.initialiseParam(L2R_L2LOSS_SVC_DUAL,CSVM);

        svmmodel.trainModel(orderedF,orderedLabels,param);

        linearClassifiers.push_back(svmmodel);
        /*for (int k=0; k<Features[0][0].size(); k++)
            cout << svmmodel.modelLinearSVM->w[k] << " ";
        cout << endl;*/

        string tmpModelPath=currPath+"/"+knownObjects[i].first+"/svmmodel";
        //svmmodel.saveModel(tmpModelPath);
    }

    cout << "trained" << endl;
    
    return true;
}

bool linearClassifierThread::trainClassifiers()
{
    lock_guard<mutex> lg(mtx);
    return trainClassifiersHelper();
}

bool linearClassifierThread::startRecognition()
{
    lock_guard<mutex> lg(mtx);
    stopAllHelper();

    if(this->linearClassifiers.size()==0)
        return false;
    
    currentState=STATE_RECOGNIZING;

    this->countBuffer.resize(bufferSize);
    this->bufferScores.resize(bufferSize);
    for (int i=0; i<bufferSize; i++)
    {
        bufferScores[i].resize(linearClassifiers.size());
        countBuffer[i].resize(linearClassifiers.size());
    }

    return true;
}

bool linearClassifierThread::forgetClassHelper(const string &className, const bool retrain)
{
    string classPath=currPath+"/"+className;
    if(yarp::os::stat(classPath.c_str()))
        return true;

    vector<string> files;
    getdir(classPath,files);

    for (int i=0; i< files.size(); i++)
    {
        if(!files[i].compare(".") || !files[i].compare(".."))
            continue;

        string feature=classPath+"/"+files[i];
        remove(feature.c_str());
    }

    bool res=(yarp::os::rmdir(classPath.c_str())==0);
    if (res && retrain)
        trainClassifiersHelper();

    return res;
}

bool linearClassifierThread::forgetClass(const string &className, const bool retrain)
{
    lock_guard<mutex> lg(mtx);
    return forgetClassHelper(className,retrain);
}

bool linearClassifierThread::forgetAll()
{
    lock_guard<mutex> lg(mtx);
    checkKnownObjects();

    for (int i=0; i<knownObjects.size(); i++)
        forgetClassHelper(knownObjects[i].first,false);

    trainClassifiersHelper();
    return true;
}


