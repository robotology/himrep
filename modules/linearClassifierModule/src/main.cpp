
#include <yarp/dev/Drivers.h>
#include "linearClassifierModule.h"

/** 
\defgroup icub_linearClassifier linearClassifier
 
The module implements a linear classifier wrapped around liblinear library. 
 
\section intro_sec Description 
This module is responsible for learning and classify feature vectors. Input features are passed, the output are the scores of SVM machines.
The module provides training methods with real-time performances.

The commands sent as bottles to the module port 
/linearClassifier/rpc are the following: 
 
(notation: [.] identifies a vocab, <.> specifies a double,
"." specifies a string) 
 
<b>SAVE</b> \n
format: [save] "class_name"  \n
action: starts to dump feature vector in the "class_name" folder.
 
<b>STOP</b>  \n
format: [stop]  \n
action: stops all the current activities, the module will be waiting for new commands.

<b>TRAIN</b>  \n
format: [train]  \n
action: starts the training process by reading all the database previously acquired via the [save] command.
 
<b>FORGET</b>  \n
format: [forget] "class"  \n
action: forgets the "class", deleting all the feature vectors in the database. If "class"="all" all the classes are forgotten. 
 
<b>LIST</b>  \n
format: [objList]  \n
action: retrieves the current classes saved in the database.

<b>CHANGE NAME</b>  \n
format: [changeName] "old_name" "new_name"  \n
action: change the current name of a known object with a new name.

<b>RECOGNIZE</b>  \n
format: [recognize]  \n
action: starts the recognition process.
 
\section lib_sec Libraries 
- YARP libraries. 

- liblinear

\section portsc_sec Ports Created 
- \e /linearClassifier/features:i receives feature vector.
 
- \e /linearClassifier/classification:o streams out the current class winner name.
 
- \e /linearClassifier/scores:o streams out the scores for the received input in the format "class" "score". 
 
- \e /linearClassifier/rpc receives requests for SIFT dump or dictionary learning. 
  
\section parameters_sec Parameters 
        
--BufferSize \e buffer
- specify the bufferSize for weighting the output scores.
 
--CSVM \e value
- specify the value for the regularization parameter used during the training.
 
--WeightedSVM \e [0 | 1]
- Useful parameter of unbalanced dataset. If 1 it weights positive examples such as their contribution is equal to the negative ones.
 
--databaseFolder \e database
- specify the database folder were feature vector will be stored.
 
 
\section tested_os_sec Tested OS
Linux, Windows 7

\author Sean Ryan Fanello
*/ 

int main(int argc, char * argv[])
{
   Network yarp;
   linearClassifierModule linearClassifierModule; 

   ResourceFinder rf;
   rf.setVerbose(true);
   rf.setDefaultConfigFile("linearClassifier.ini"); 
   rf.setDefaultContext("himrep");
   rf.configure(argc, argv);

   linearClassifierModule.runModule(rf);

   return 0;
}
