#include <iostream>
#include <string>

#include <yarp/os/all.h>
#include <yarp/sig/all.h>
#include <yarp/math/Math.h>

#include <iCub/ctrl/math.h>

#include "SVMLinear.h"

#ifdef _WIN32
    #include "win_dirent.h"
#else
    #include "dirent.h"
#endif

#define STATE_DONOTHING         0
#define STATE_SAVING            1
#define STATE_RECOGNIZING       2

using namespace std; 
using namespace yarp::os; 
using namespace yarp::sig;
using namespace yarp::math;


class linearClassifierThread : public Thread
{
private:
    string inputFeatures;
    string outputPortName;
    string outputScorePortName;

    string currPath;
    string pathObj;
    Port *commandPort;
    BufferedPort<Bottle> featuresPort;
    BufferedPort<Bottle> outputPort;
    Port scorePort;
    Mutex mutex;
    fstream objFeatures;

    vector<pair<string,vector<string> > > knownObjects;
    vector<SVMLinear> linearClassifiers;
    vector<int> datasetSizes;
    vector<vector<vector<double> > > Features;
    vector<vector<double > > bufferScores;
    vector<vector<int > > countBuffer;
    int bufferSize;
    double CSVM;
    int useWeightedSVM;

    int currentState;

    void checkKnownObjects();
    int getdir(const string &dir, vector<string> &files);
    void stopAllHelper();
    bool trainClassifiersHelper();
    bool forgetClassHelper(const string &className, const bool retrain);

public:

    linearClassifierThread(yarp::os::ResourceFinder &rf,Port* commPort);
    void prepareObjPath(const string &objName);
    bool threadInit();
    void threadRelease();
    void run(); 
    void onStop();
    void createFullPath(const char * path);
    void stopAll();
    bool loadFeatures();
    bool trainClassifiers();
    bool startRecognition();
    bool forgetClass(const string &className, const bool retrain);
    bool forgetAll();
    bool getClassList(Bottle &b);
    bool changeName(const string &old_name, const string &new_name);
};
