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
#include <numeric>
#include <utility>
#include <mutex>

#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/PortReport.h>
#include <yarp/os/Stamp.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>

#include <yarp/cv/Cv.h>

#include <yarp/math/Math.h>

#include "CaffeFeatExtractor.hpp"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::cv;
using namespace yarp::math;

#define CMD_HELP                    yarp::os::createVocab32('h','e','l','p')
#define DUMP_CODE                   yarp::os::createVocab32('d','u','m','p')
#define DUMP_STOP                   yarp::os::createVocab32('s','t','o','p')

class CaffeCoderPort: public BufferedPort<Image>
{
private:

    // Resource Finder and module options

    ResourceFinder                &rf;

    string                        contextPath;

    bool                          dump_code;

    double                        rate;
    double                        last_read;

    // Data (common to all methods)

    ::cv::Mat                     matImg;

    Port                          port_out_img;
    Port                          port_out_code;

    FILE                          *fout_code;

    mutex                         mtx;

    list<double>                  time_measurements_prep;
    list<double>                  time_measurements_net;
    int                           time_avg_window;

    // Data (specific for each method - instantiate only those are needed)

    CaffeFeatExtractor<float>    *caffe_extractor;

    void onRead(Image &img)
    {
    	// Read at specified rate
        if (Time::now() - last_read < rate)
            return;

        lock_guard<mutex> lg(mtx);

        // If something arrived...
        if (img.width()>0 && img.height()>0)
        {
            // Convert the image
            ImageOf<PixelRgb> tmp_img;
            tmp_img.copy(img);
            matImg = toCvMat(tmp_img);

            // Extract the feature vector

            std::vector<float> codingVecFloat;
            float times[2];
            if (!caffe_extractor->extract_singleFeat_1D(matImg, codingVecFloat, times))
            {
                std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): failed..." << std::endl;
                return;
            }
            std::vector<double> codingVec(codingVecFloat.begin(), codingVecFloat.end());

            if (caffe_extractor->timing)
            {
                time_measurements_prep.push_back(times[0]);
                time_measurements_net.push_back(times[1]);

                while (time_measurements_prep.size()>time_avg_window)
                {
                    time_measurements_prep.pop_front();
                    time_measurements_net.pop_front();
                }

                double prep_sum = std::accumulate(time_measurements_prep.begin(), time_measurements_prep.end(), 0.0);
                double prep_mean = prep_sum / time_measurements_prep.size();
                double prep_sq_sum = std::inner_product(time_measurements_prep.begin(), time_measurements_prep.end(), time_measurements_prep.begin(), 0.0);
                double prep_stdev = std::sqrt(prep_sq_sum / time_measurements_prep.size() - prep_mean * prep_mean);

                double net_sum = std::accumulate(time_measurements_net.begin(), time_measurements_net.end(), 0.0);
                double net_mean = net_sum / time_measurements_net.size();
                double net_sq_sum = std::inner_product(time_measurements_net.begin(), time_measurements_net.end(), time_measurements_net.begin(), 0.0);
                double net_stdev = std::sqrt(net_sq_sum / time_measurements_net.size() - net_mean * net_mean);

                std::cout << "PREP: " << prep_mean << " - " << prep_stdev << endl;
                std::cout << "NET: " << net_mean << " - " << net_stdev << endl;

                std::cout << "time_avg_window = " << time_measurements_prep.size() << endl;
            }

            // Dump if required
            if (dump_code)
            {
                fwrite (&codingVec[0], sizeof(double), codingVec.size(), fout_code);
            }

            Stamp stamp;
            this->getEnvelope(stamp);

            if(port_out_code.getOutputCount())
            {
                port_out_code.setEnvelope(stamp);
                Vector codingYarpVec(codingVec.size(), &codingVec[0]);
                port_out_code.write(codingYarpVec);
            }

            if(port_out_img.getOutputCount())
            {
                port_out_img.write(img);
            }
        }
    }

public:

    CaffeCoderPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
    {
        // Resource Finder and module options
        contextPath = rf.getHomeContextPath().c_str();

        // Data initialization (specific for Caffe method)

        // Binary file (.caffemodel) containing the network's weights
        string caffemodel_file = rf.check("caffemodel_file", Value("bvlc_googlenet.caffemodel")).asString().c_str();
        cout << "Setting .caffemodel file to " << caffemodel_file << endl;

        // Text file (.prototxt) defining the network structure
        string prototxt_file = rf.check("prototxt_file", Value(contextPath + "/bvlc_googlenet_val_cutpool5.prototxt")).asString().c_str();
        cout << "Setting .prototxt file to " << prototxt_file << endl;

        // Name of blobs to be extracted
        string blob_name = rf.check("blob_name", Value("pool5/7x7_s1")).asString().c_str();
        cout << "Setting blob_name to " << blob_name << endl;

        bool timing = rf.check("timing");
        if (timing)
        {
            time_avg_window = rf.check("timing",Value("1000")).asInt32();
        }
        else
        {
            time_avg_window = -1;
        }

        // Compute mode and eventually GPU ID to be used
        int device_id;
        string compute_mode;

        #ifdef HAS_CUDA
            compute_mode = rf.check("compute_mode", Value("GPU")).asString();
            device_id = rf.check("device_id", Value(0)).asInt32();
        #else
            compute_mode = "CPU";
            device_id = -1;
        #endif

        int resizeWidth = rf.check("resizeWidth", Value(256)).asFloat64();
        int resizeHeight = rf.check("resizeHeight", Value(256)).asFloat64();

        caffe_extractor = NULL;
        caffe_extractor = new CaffeFeatExtractor<float>(caffemodel_file,
                prototxt_file, resizeWidth, resizeHeight,
                blob_name,
                compute_mode,
                device_id,
                timing);

        // Data (common to all methods)
        string name = rf.find("name").asString().c_str();

        port_out_img.open(("/"+name+"/img:o").c_str());
        port_out_code.open(("/"+name+"/code:o").c_str());

        BufferedPort<Image>::useCallback();

        rate = rf.check("rate",Value(0.0)).asFloat64();
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
        lock_guard<mutex> lg(mtx);

        port_out_code.interrupt();
        port_out_img.interrupt();

        BufferedPort<Image>::interrupt();
    }

    void resume()
    {
        lock_guard<mutex> lg(mtx);

        port_out_code.resume();
        port_out_img.resume();

        BufferedPort<Image>::resume();
    }

    void close()
    {
        lock_guard<mutex> lg(mtx);

        if (dump_code)
        {
            fclose(fout_code);
        }

        port_out_code.close();
        port_out_img.close();

        BufferedPort<Image>::close();
    }

    bool execReq(const Bottle &command, Bottle &reply)
    {
        switch(command.get(0).asVocab32())
        {
        case(CMD_HELP):
            {
            reply.clear();
            reply.add(Value::makeVocab32("many"));
            reply.addString("[dump] [path-to-file] [a] to start dumping the codes in the context directory. Use 'a' for appending.");
            reply.addString("[stop] to stop dumping.");
            return true;
            }

        case(DUMP_CODE):
            {
            lock_guard<mutex> lg(mtx);

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

            return true;
            }

        case(DUMP_STOP):
            {
            lock_guard<mutex> lg(mtx);

            dump_code = false;
            fclose(fout_code);
            reply.addString("Stopped code dump.");

            return true;
            }

        default:
            return false;
        }
    }
};


class CaffeCoderModule: public RFModule
{
protected:
    CaffeCoderPort         *caffePort;
    Port                   rpcPort;

public:

    CaffeCoderModule()
    {
        caffePort=NULL;
    }

    bool configure(ResourceFinder &rf)
    {
        string name = rf.find("name").asString().c_str();

        caffePort = new CaffeCoderPort(rf);

        caffePort->open(("/"+name+"/img:i").c_str());

        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    bool interruptModule()
    {
        if (caffePort!=NULL)
            caffePort->interrupt();

        rpcPort.interrupt();

        return true;
    }

    bool close()
    {
        if(caffePort!=NULL)
        {
            caffePort->close();
            delete caffePort;
        }

        rpcPort.close();

        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        if (caffePort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    double getPeriod()    { return 1.0;  }

    bool updateModule()
    {
        //caffePort->update();

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
    rf.setDefaultConfigFile("caffeCoder.ini");
    rf.configure(argc,argv);
    rf.setDefault("name","caffeCoder");

    CaffeCoderModule mod;
    return mod.runModule(rf);
}
