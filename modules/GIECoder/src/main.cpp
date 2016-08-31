/*
 * Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Giulia Pasquale
 * email:  giulia.pasquale@iit.it
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

// General includes
#include <cstdio>
#include <cstdlib> // getenv
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

// OpenCV
#include <opencv2/opencv.hpp>

#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Semaphore.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/PortReport.h>
#include <yarp/os/Stamp.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>

#include <yarp/math/Math.h>

#include "GIEFeatExtractor.h"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;

#define CMD_HELP                    VOCAB4('h','e','l','p')
#define DUMP_CODE                   VOCAB4('d','u','m','p')
#define DUMP_STOP                   VOCAB4('s','t','o','p')

class GIECoderPort: public BufferedPort<Image>
{
private:

    // Resource Finder and module options

    ResourceFinder                &rf;

    string                        contextPath;

    bool                          dump_code;

    double                        rate;
    double                        last_read;

    // Data (common to all methods)

    cv::Mat                       matImg;

    Port                          port_out_img;
    Port                          port_out_code;

    FILE                          *fout_code;

    Semaphore                     mutex;

    // Data (specific for each method - instantiate only those are needed)

    GIEFeatExtractor              *gie_extractor;

    void onRead(Image &img)
    {

    	// Read at specified rate
        if (Time::now() - last_read < rate)
            return;

        mutex.wait();

        // If something arrived...
        if (img.width()>0 && img.height()>0)
        {

            // Convert the image and check that it is continuous

            cv::Mat tmp_mat = cv::cvarrToMat((IplImage*)img.getIplImage());
            cv::cvtColor(tmp_mat, matImg, CV_RGB2BGR);

            // Extract the feature vector

            std::vector<float> codingVecFloat;
            float times[2];
            if (!gie_extractor->extract_singleFeat_1D(matImg, codingVecFloat, times))
            {
                std::cout << "GIEFeatExtractor::extract_singleFeat_1D(): failed..." << std::endl;
                return;
            }
            std::vector<double> codingVec(codingVecFloat.begin(), codingVecFloat.end());

            if (gie_extractor->timing)
            {
                std::cout << times[0] << ": PREP " << times[1] << ": NET" << std::endl;
            }

            if (dump_code)
            {
                fwrite (&codingVec[0], sizeof(double), codingVec.size(), fout_code);
            }

            Stamp stamp;
            this->getEnvelope(stamp);

            if (port_out_code.getOutputCount())
            {
                port_out_code.setEnvelope(stamp);
                yarp::sig::Vector codingYarpVec(codingVec.size(), &codingVec[0]);
                port_out_code.write(codingYarpVec);
            }

            if (port_out_img.getOutputCount())
            {
                port_out_img.write(img);
            }
        }

        mutex.post();

    }

public:

    GIECoderPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
    {

        // Resource Finder and module options

        contextPath = rf.getHomeContextPath().c_str();

        // Data initialization (specific for Caffe method)

        // Binary file (.caffemodel) containing the network's weights
        string caffemodel_file = rf.check("caffemodel_file", Value("/usr/local/src/robot/GIE/models/bvlc_googlenet/bvlc_googlenet.caffemodel")).asString().c_str();
        cout << "Setting .caffemodel file to " << caffemodel_file << endl;
           
        // Text file (.prototxt) defining the network structure
        string prototxt_file = rf.check("prototxt_file", Value("/usr/local/src/robot/GIE/models/bvlc_googlenet/deploy.prototxt")).asString().c_str();
        cout << "Setting .prototxt file to " << prototxt_file << endl;

        // Name of blobs to be extracted
        string blob_name = rf.check("blob_name", Value("pool5/7x7_s1")).asString().c_str();

        // Boolean flag for timing or not the feature extraction
        bool timing = rf.check("timing",Value(false)).asBool();
    
        string  binaryproto_meanfile = "";
        float meanR = -1, meanG = -1, meanB = -1;
        int resizeWidth = -1, resizeHeight = -1;
        if (rf.find("binaryproto_meanfile").isNull() && rf.find("meanR").isNull())
        {
            cout << "ERROR: missing mean info!!!!!" << endl;
        }
        else if (rf.find("binaryproto_meanfile").isNull())
        {
            meanR = rf.check("meanR", Value(123)).asDouble();
            meanG = rf.check("meanG", Value(117)).asDouble();
            meanB = rf.check("meanB", Value(104)).asDouble();
            resizeWidth = rf.check("resizeWidth", Value(256)).asDouble();
            resizeHeight = rf.check("resizeHeight", Value(256)).asDouble();
            std::cout << "Setting mean to " << " R: " << meanR << " G: " << meanG << " B: " << meanB << std::endl;
            std::cout << "Resizing anysotropically to " << " W: " << resizeWidth << " H: " << resizeHeight << std::endl;
            
        } 
        else if (rf.find("meanR").isNull())
        {
            binaryproto_meanfile = rf.check("binaryproto_meanfile", Value("/usr/local/src/robot/GIE/data/ilsvrc12/imagenet_mean.binaryproto")).asString().c_str();
            cout << "Setting .binaryproto file to " << binaryproto_meanfile << endl;
        }
        else
        {
            std::cout << "ERROR: need EITHER mean file (.binaryproto) OR mean pixel values!" << std::endl;
        }

        gie_extractor = new GIEFeatExtractor(caffemodel_file, binaryproto_meanfile, meanR, meanG, meanB,
                prototxt_file, resizeWidth, resizeHeight,
                blob_name,
                timing);
	    if( !gie_extractor )
	    {
		    cout << "Failed to initialize GIEFeatExtractor" << endl;
	    }

        // Data (common to all methods)

        string name = rf.find("name").asString().c_str();

        port_out_img.open(("/"+name+"/img:o").c_str());
        port_out_code.open(("/"+name+"/code:o").c_str());

        BufferedPort<Image>::useCallback();

        rate = rf.check("rate",Value(0.0)).asDouble();
        last_read = 0.0;

        dump_code = rf.check("dump_code");
        if(dump_code)
        {
            string code_path = rf.check("dump_code",Value("codes.bin")).asString().c_str();
            code_path = contextPath + "/" + code_path;
            string code_write_mode = rf.check("append")?"wb+":"wb";

            fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
        }

    }

    void interrupt()
    {
        mutex.wait();

        port_out_code.interrupt();
        port_out_img.interrupt();

        BufferedPort<Image>::interrupt();

        mutex.post();
    }

    void resume()
    {
        mutex.wait();

        port_out_code.resume();
        port_out_img.resume();

        BufferedPort<Image>::resume();

        mutex.post();
    }

    void close()
    {
        mutex.wait();

        if (dump_code)
        {
            fclose(fout_code);
        }

        port_out_code.close();
        port_out_img.close();

        delete gie_extractor;

        BufferedPort<Image>::close();

        mutex.post();
    }

    bool execReq(const Bottle &command, Bottle &reply)
    {
        switch(command.get(0).asVocab())
        {
        case(CMD_HELP):
            {
            reply.clear();
            reply.add(Value::makeVocab("many"));
            reply.addString("[dump] [path-to-file] [a] to start dumping the codes in the context directory. Use 'a' for appending.");
            reply.addString("[stop] to stop dumping.");
            return true;
            }

        case(DUMP_CODE):
            {
            mutex.wait();

            dump_code = true;
            string code_path;
            string code_write_mode;

            if (command.size()==1)
            {
                code_path = contextPath + "/codes.bin";
                code_write_mode="wb";
            }
            else if (command.size()==2)
            {
                if (strcmp(command.get(1).asString().c_str(),"a")==0)
                {
                    code_write_mode="wb+";
                    code_path = contextPath + "/codes.bin";
                } else
                {
                    code_write_mode="wb";
                    code_path = command.get(1).asString().c_str();
                }
            } else if (command.size()==3)
            {
                code_write_mode="wb+";
                code_path = command.get(2).asString().c_str();
            }

            fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
            reply.addString("Start dumping codes...");

            mutex.post();
            return true;
            }

        case(DUMP_STOP):
            {
            mutex.wait();

            dump_code = false;
            fclose(fout_code);
            reply.addString("Stopped code dump.");

            mutex.post();

            return true;
            }

        default:
            return false;
        }
    }

};


class GIECoderModule: public RFModule
{
protected:
    GIECoderPort         *GIEPort;
    Port                   rpcPort;

public:

    GIECoderModule()
{
        GIEPort=NULL;
}

    bool configure(ResourceFinder &rf)
    {

        string name = rf.find("name").asString().c_str();

        Time::turboBoost();

        GIEPort = new GIECoderPort(rf);

        GIEPort->open(("/"+name+"/img:i").c_str());

        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    bool interruptModule()
    {
        if (GIEPort!=NULL)
            GIEPort->interrupt();

        rpcPort.interrupt();

        return true;
    }

    bool close()
    {
        if(GIEPort!=NULL)
        {
            GIEPort->close();
            delete GIEPort;
        }

        rpcPort.close();

        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        if (GIEPort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    double getPeriod()    { return 1.0;  }

    bool updateModule()
    {
        //GIEPort->update();

        return true;
    }

};


int main(int argc, char *argv[])
{
    Network yarp;

    if (!yarp.checkNetwork())
        return 1;

    ResourceFinder rf;

    rf.setVerbose(true);

    rf.setDefaultContext("himrep");
    rf.setDefaultConfigFile("GIECoder_googlenet.ini");

    rf.configure(argc,argv);

    rf.setDefault("name","GIECoder");

    GIECoderModule mod;

    return mod.runModule(rf);
}

