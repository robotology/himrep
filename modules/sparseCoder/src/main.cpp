/* 
 * Copyright (C) 2013 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Sean Ryan Fanello, Carlo Ciliberto
 * email:  sean.fanello@iit.it carlo.ciliberto@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
*/

/** 
\defgroup icub_sparseCoder sparseCoder
 
The module performs a sequence of coding and pooling operator and retrieves a hierarchical image representation.
 
\section intro_sec Description 
This module is responsible for coding images into feature vectors. To this end,
we extract local appearance descriptors (sparse or dense). These descriptors are coded
using a learned dictionary and the pooled together via a spatial pyramid representation.
 
The commands sent as bottles to the module port 
/sparseCoder/rpc are the following: 
 
(notation: [.] identifies a vocab, <.> specifies a double,
"." specifies a string) 
 
<b>HELP</b> 
format: [help]
action: it prints the available commands.
 
 
<b>DUMP_SIFT</b> 
format: [dump] "a"
action: starts to dump local descriptors that can be used for dictionary learning. "a" is an optional parameters that stands for "append" in case you want to keep old descriptors previously dumped.

<b>DICTIONARY_LEARNING</b> 
format: [learn] <dictionarySize> "usePCA" <dimPCA>
action: starts to learn a dictionary from previously dumped local descriptors. <dictionarySize> is the number of atoms of the dictionary, "usePCA" performs standard PCA analysis on low-level descriptors before the encoding and <dimPCA> indicates the number of relevant eigenvectors that you want use for the projection.
 
\section lib_sec Libraries 
- YARP libraries. 

- OpenCV 2.2 libraries.

\section portsc_sec Ports Created 
- \e /sparseCoder/img:i receives the image acquired from the 
  camera.
 
- \e /sparseCoder/img:o streams out the image containing 
  the extracted local descriptors.
 
- \e /sparseCoder/code:o streams out the vector containing 
  the hierarchical image representation. 
 
- \e /sparseCoder/rpc receives requests for SIFT dump or dictionary learning. 
  
\section parameters_sec Parameters 
        
--dictionary_file \e file
- specify the dictionary file.ini where atoms are stored.
 
--coding_mode \e coding
- specify the coding method used. Possible values are "BOW" for Bag of Words, 
  "SC" for Sparse Coding, "BCE" for Best Codes Entries.
 
--nearest_neighbor \e knn
- specify the number of nearest neighbors used for the encoding with BCE.
 
--dense_descriptors \e [1|0] 
- specify is a regular grid of dense descriptors must be used (1) or sparse keypoints must be detected (0).
 
--grid_scale \e scale
- specify the scale of local descriptors extracted from a dense grid.
 
--grid_step \e step
- specify the sampling step among descriptors in a dense grid.

\section tested_os_sec Tested OS
Linux

\author Sean Ryan Fanello, Carlo Ciliberto 
*/ 

#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/PortReport.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>

#include <yarp/cv/Cv.h>

#include <yarp/math/Math.h>
#include <yarp/math/Rand.h>

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>
#include <deque>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>
#include <mutex>

#include "SiftGPU_Extractor.h"
#include "DictionaryLearning.h"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::cv;
using namespace yarp::math;

#define CMD_HELP                    yarp::os::createVocab('h','e','l','p')
#define DUMP_SIFT                   yarp::os::createVocab('d','u','m','p')
#define DUMP_STOP                   yarp::os::createVocab('s','t','o','p')
#define LEARN_DICT                  yarp::os::createVocab('l','e','a','r')

#define CODE_MODE_SC                0
#define CODE_MODE_BOW               1
#define CODE_MODE_BCE               2


class SparseCoderPort: public BufferedPort<Image>
{
private:
    ResourceFinder                      &rf;

    bool                                verbose;
    bool                                help;

    vector<SiftGPU::SiftKeypoint>       keypoints;
    vector<float>                       descriptors;

    Port                                port_out_code;
    Port                                port_out_img;

    SiftGPU_Extractor                   siftGPU_extractor;

    int                                 grid_step;
    int                                 grid_scale;

    DictionaryLearning                  *sparse_coder;
    IplImage                            *ipl;

    mutex                               mtx;
    
    bool                                no_code;
    int                                 dense;
    bool                                dump_sift;
    FILE                                *fout_sift;

    double                              rate;
    double                              last_read;

    int                                 code_mode;
    int                                 pyramidLevels;

    string                              contextPath;


    virtual void onRead(Image &img)
    {
        //read at specified rate
        if(Time::now()-last_read<rate)
            return;

        lock_guard<mutex> lg(mtx);
        if(ipl==NULL || ipl->width!=img.width() || ipl->height!=img.height())
        {
            if(ipl!=NULL)
            {
                cvReleaseImage(&ipl);
            }
            ipl=cvCreateImage(cvSize(img.width(),img.height()),IPL_DEPTH_8U,1);
            siftGPU_extractor.setDenseGrid(ipl,grid_step,grid_scale);
        }

        cv::cvtColor(toCvMat(img),cv::cvarrToMat(ipl),CV_RGB2GRAY);
        
        //cvSmooth(ipl,ipl);  
        if(dense)
            siftGPU_extractor.extractDenseSift(ipl,&keypoints,&descriptors);
        else
            siftGPU_extractor.extractSift(ipl,&keypoints,&descriptors);
       
        if(dump_sift)
        {
            for(int i=0; i<keypoints.size(); i++)
            {
                for(int j=0; j<128; j++)
                     fprintf(fout_sift,"%f ",descriptors[i*128+j]);
                fprintf(fout_sift,"\n");
            }
        }
        //code the input vector
        if(!no_code)
        {
            vector<Vector> features(keypoints.size());
            Vector coding;
            for(unsigned int i=0; i<keypoints.size(); i++)
            {
                features[i].resize(128);
                for(unsigned int j=0; j<128; j++)
                    features[i][j]=descriptors[i*128+j];
            }
            
            switch(code_mode)
            {
                case CODE_MODE_SC:
                {
                    sparse_coder->maxPooling(features,coding,keypoints,pyramidLevels,ipl->width, ipl->height);
                    break;
                }
                case CODE_MODE_BCE:
                {
                    sparse_coder->maxPooling(features,coding,keypoints,pyramidLevels,ipl->width, ipl->height);
                    break;
                }                
                case CODE_MODE_BOW:
                {
                    sparse_coder->avgPooling(features,coding,keypoints,pyramidLevels,ipl->width, ipl->height);
                    break;
                }
            }
            
            
            if(port_out_code.getOutputCount())
            {
                port_out_code.write(coding);
            }
            if(port_out_img.getOutputCount())
            {
                for(unsigned int i=0; i<keypoints.size(); i++)
                {   
                    int x = cvRound(keypoints[i].x);
                    int y = cvRound(keypoints[i].y);
                    cv::Mat imgMat=toCvMat(img);
                    cv::circle(imgMat,cvPoint(x,y),3,cvScalar(0,0,255),-1);
                }
                port_out_img.write(img);
            }
        }
    }


public:
   SparseCoderPort(ResourceFinder &_rf)
       :BufferedPort<Image>(),rf(_rf)
   {
        ipl=NULL;
        help=false;
        verbose=rf.check("verbose");

        grid_step=rf.check("grid_step",Value(8)).asInt();
        grid_scale=rf.check("grid_scale",Value(1)).asInt();

        contextPath=rf.getHomeContextPath().c_str();
        string dictionary_name=rf.check("dictionary_file",Value("dictionary_bow.ini")).asString().c_str();

        string dictionary_path=rf.findFile(dictionary_name);
        if(dictionary_path=="")
            dictionary_path=contextPath+"/"+dictionary_name;
        string dictionary_group=rf.check("dictionary_group",Value("DICTIONARY")).asString().c_str();

        no_code=rf.check("no_code");
        dump_sift=rf.check("dump_sift");

        if(dump_sift)
        {
            string sift_path=rf.check("dump_sift",Value("sift.txt")).asString().c_str();
            sift_path=contextPath+"/"+sift_path;
            string sift_write_mode=rf.check("append")?"a":"w";

            fout_sift=fopen(sift_path.c_str(),sift_write_mode.c_str());
        }

        rate=rf.check("rate",Value(0.0)).asDouble();
        dense=rf.check("useDense",Value(1)).asInt();
        int knn=rf.check("KNN",Value(5)).asInt();
        last_read=0.0;

        pyramidLevels=rf.check("PyramidLevels",Value(3)).asInt();

            if(dense)
                  fprintf(stdout,"Step: %d Scale: %d Pyramid: %d Using Dense SIFT Grid\n",grid_step, grid_scale, pyramidLevels);
            else
                  fprintf(stdout,"Step: %d Scale: %d Pyramid: %d Using Sparse SIFTs \n",grid_step, grid_scale, pyramidLevels);
                                        
        string code_mode_string=rf.check("code_mode",Value("SC")).asString().c_str();          
                                      
        sparse_coder=NULL;
        sparse_coder=new DictionaryLearning(dictionary_path,dictionary_group,code_mode_string,knn);

        
        
        //set all chars to lower case
        for(int i=0; i<code_mode_string.size(); i++)
            code_mode_string[i] = std::toupper((unsigned char)code_mode_string[i]);

        fprintf(stdout,"%s\n",code_mode_string.c_str());
        
        if(code_mode_string=="SC")
            code_mode=CODE_MODE_SC;
        if(code_mode_string=="BCE")
            code_mode=CODE_MODE_BCE;
        if(code_mode_string=="BOW")
            code_mode=CODE_MODE_BOW;

        string name=rf.find("name").asString().c_str();

        port_out_img.open(("/"+name+"/img:o").c_str());
        port_out_code.open(("/"+name+"/code:o").c_str());
        BufferedPort<Image>::useCallback();
   }

   virtual void interrupt()
   {
        lock_guard<mutex> lg(mtx);
        port_out_code.interrupt();
        port_out_img.interrupt();
        BufferedPort<Image>::interrupt();
   }

   virtual void resume()
   {
        lock_guard<mutex> lg(mtx);
        port_out_code.resume();
        port_out_img.resume();
        BufferedPort<Image>::resume();
   }

   virtual void close()
   {
        lock_guard<mutex> lg(mtx);
        if(ipl!=NULL)
            cvReleaseImage(&ipl);

        if(dump_sift)
            fclose(fout_sift);

        //some closure :P
        if(sparse_coder!=NULL)
            delete sparse_coder;

        port_out_code.close();
        port_out_img.close();
        BufferedPort<Image>::close();
   }


   bool execReq(const Bottle &command, Bottle &reply)
   {
       switch(command.get(0).asVocab())
       {
           case(CMD_HELP):
           {
                reply.clear();
                if(!help)
                {
                    reply.addString("There's no help in this life. You oughta do everything on your own");
                    help=true;
                } else
                {
                    reply.add(Value::makeVocab("many"));
                    reply.addString("Ok joking.. here the help!");                    
                    reply.addString("[dump] [a] for starting the SIFT dumping in the context directory.. use 'a' for appending");
                    reply.addString("[stop] for stopping the SIFT dumping in the context directory.. ");
                    reply.addString("[learn] [dictionarySize] [usePCA] [dimPCA] for learning the dictionary from a SIFT file previously dumped..");
                }
                
                return true;
           }
           case(DUMP_SIFT):
           {
                lock_guard<mutex> lg(mtx);
                dump_sift=true; 
                string sift_path="sift.txt";
                sift_path=contextPath+"/"+sift_path;

                string sift_write_mode;
                if(command.size()==1)
                    sift_write_mode="w";
                else
                    sift_write_mode=command.get(1).asString().c_str();

                fout_sift=fopen(sift_path.c_str(),sift_write_mode.c_str());
                reply.addString("Starting to dump SIFTs...");
                return true;

           }
           case(DUMP_STOP):
           {
                lock_guard<mutex> lg(mtx);
                dump_sift=false;                
                fclose(fout_sift);
                reply.addString("Stopped SIFT Dump.");
                return true;
           }
           case(LEARN_DICT):
           {
                lock_guard<mutex> lg(mtx);
                //string sift_path="sift.txt";
                //sift_path=contextPath+"/"+sift_path;
                string sift_path=rf.findFile("sift.txt").c_str();
                string dictionary_path="newDictionary.ini";
                dictionary_path=contextPath+"/"+dictionary_path;
                reply.addString("Dictionary Learned.");
                int dictSize;
                if(command.size()==1)
                   dictSize=512;
                else
                   dictSize=command.get(1).asInt();

                bool usePCA=command.size()>2;

                int dimPCA;
                if(command.size()>3)
                {
                    dimPCA=command.get(3).asInt();
                }else
                {
                    if(usePCA)
                        dimPCA=80;
                }
                sparse_coder->learnDictionary(sift_path, dictSize,usePCA,dimPCA);
                sparse_coder->saveDictionary(dictionary_path);
                return true;
           }
           default:
               return false;
       }
   }


};


class SparseCoderModule: public RFModule
{
protected:
    SparseCoderPort         *scPort;
    Port                    rpcPort;

public:
    SparseCoderModule()
    {
        scPort=NULL;
    }

    virtual bool configure(ResourceFinder &rf)
    {
        string name=rf.find("name").asString().c_str();

        scPort=new SparseCoderPort(rf);
        scPort->open(("/"+name+"/img:i").c_str());
        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    virtual bool close()
    {
        if(scPort!=NULL)
        {
            scPort->interrupt();
            scPort->close();
            delete scPort;
        }
        
        rpcPort.interrupt();
        rpcPort.close();

        return true;
    }

    virtual bool respond(const Bottle &command, Bottle &reply)
    {
        if(scPort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    virtual double getPeriod()    { return 1.0;  }
    virtual bool   updateModule()
    {
        //scPort->update();

        return true;
    }

};




int main(int argc, char *argv[])
{
   Network yarp;

   if (!yarp.checkNetwork())
       return 1;

   ResourceFinder rf;
   rf.setDefaultContext("himrep");
   rf.setDefaultConfigFile("sparseCoder.ini");
   rf.configure(argc,argv);
   rf.setDefault("name","sparseCoder");
   SparseCoderModule mod;

   return mod.runModule(rf);
}

