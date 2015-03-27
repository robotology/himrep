/* 
 * Copyright (C) 2014 iCub Facility - Istituto Italiano di Tecnologia
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

/** 
\defgroup icub_caffeCoder caffeCoder

The module computes a representation of the input image using 
Caffe library.

\section intro_sec Description 
The input image is passed through a feed-forward Convolutional Network model 
(that must have been previously learned). The image representation 
can be extracted as the output of any layer in the network. 
Usually the latest convolutional layer is chosen and the output is a 
one-dimensional vector, than can be fed directly to a (linear) classifier.

The commands sent as bottles to the module port 
/caffeCoder/rpc are the following: 

(notation: [.] identifies a vocab, <.> specifies a double,
"." specifies a string) 

<b>HELP</b> 
format: [help]
action: it prints the available commands.


<b>DUMP_CODE</b>
format: [dump] "a"
action: starts to save features in a binary file. "a" is optional and stands for "append".

\section lib_sec Libraries 
- YARP libraries. 

- OpenCV 2.4 libraries.

- Caffe libraries

- CUDA libraries

\section portsc_sec Ports Created 
- \e /caffeCoder/img:i receives an image.

- \e /caffeCoder/img:o streams out the image.

- \e /caffeCoder/code:o streams out the vector containing 
the hierarchical image representation.

- \e /caffeCoder/rpc receives requests.

\section parameters_sec Parameters 

--pretrained_binary_proto_file \e file
- Absolute path to the .caffemodel file containing the weights of the learned model to be used.
If absent, it is set to '$Caffe_ROOT/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
if Caffe_ROOT env variable exists, otherwise it is set to empty string.

feature_extraction_proto_file \e file
- File name of the .prototxt file containing the network definition.
If absent, it is set to 'imagenet_val_cutfc6.prototxt'.

extract_features_blob_names \e layer_id
Tag (see the .prototxt definition) of the layer at which we want to extract the features.
If absent, it is set to 'fc6'.
It can be more than one layer tags (separated by commas without spaces accordingly with Caffe's standard). 
However, in this case the module (see main.cpp) must call a different function among those exposed by the 
CaffeFeatExtractor wrapper. 

compute_mode \e [GPU|gpu|CPU|cpu]
- Argument passed to Caffe initialization. Set to 'GPU' if absent.
Set it accordingly with your compilation of Caffe.
However, it must be noted that, at present, the CaffeFeatExtractor wrapper uses CUDA events for timing the extraction.
This is the only direct dependency of this module on CUDA at the moment.
If you want to run Caffe on the CPU and do not have CUDA installed, 
you have to remove this dependency from CaffeFeatExtractor.hpp (removing CUDA includes and 
deleting the timing code) and also remove the CUDA package from the CMakeLists.txt inside caffeCoder/src.

device_id \e dev_id 
- Argument passed to Caffe initialization. Device chosen by Caffe to run the computations. 
Used if compute_mode is set to GPU. Set to '0' if absent.

timing \e [true|false]
- Set to 'true' if you desire to read in the stdout the time to extract the features, to 'false' otherwise.

\section tested_os_sec Tested OS
Linux - Ubuntu 14.04

\author Giulia Pasquale
 */

// YARP
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

// OpenCV
#include <opencv/highgui.h>
#include <opencv/cv.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// General includes
#include <stdio.h>
#include <stdlib.h> // getenv
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "CaffeFeatExtractor.hpp"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;

#define CMD_HELP                    VOCAB4('h','e','l','p')
#define DUMP_CODE                   VOCAB4('d','u','m','p')
#define DUMP_STOP                   VOCAB4('s','t','o','p')

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

	cv::Mat				  		  matImg;

	Port                		  port_out_img;
	Port                		  port_out_code;

	FILE                          *fout_code;

	Semaphore                     mutex;

	// Data (specific for each method - instantiate only those are needed)

	CaffeFeatExtractor<double>    *caffe_extractor;

	void onRead(Image &img)
	{

		// Read at specified rate
		if (Time::now() - last_read < rate)
			return;

		mutex.wait();

		// If something arrived...
		if(img.width()>0 && img.height()>0)
		{

			// Convert the image

			cv::cvtColor(cv::Mat((IplImage*)img.getIplImage()), matImg, CV_RGB2BGR);

			// Extract the feature vector

			std::vector<double> codingVec;
			float msecPerImage = caffe_extractor->extract_singleFeat_1D(matImg, codingVec);

			if (caffe_extractor->timing)
			{
				cout << msecPerImage << " msec" << endl;
			}

			// Dump if required
			if(dump_code)
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

		mutex.post();

	}

public:

	CaffeCoderPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
	{

		// Resource Finder and module options

		contextPath = rf.getHomeContextPath().c_str();

		// Data initialization (specific for Caffe method)

		// Binary file (.caffemodel) containing the network's weights
		string pretrained_binary_proto_file;
		if (rf.find("pretrained_binary_proto_file").isNull())
		{
			pretrained_binary_proto_file = "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel";

			string Caffe_ROOT = string( getenv("Caffe_ROOT") );
			if (Caffe_ROOT!="")
			{
				pretrained_binary_proto_file = Caffe_ROOT + "/" + pretrained_binary_proto_file;
				cout << "Setting .caffemodel file to " << pretrained_binary_proto_file << endl;
			} else
			{
				cout << "Empty env variable 'Caffe_ROOT' and missing 'pretrained_binary_proto_file' in .ini, setting .caffemodel file to ''" << endl;
				pretrained_binary_proto_file = "";
			}
		}
		else
		{
			pretrained_binary_proto_file = rf.find("pretrained_binary_proto_file").asString().c_str();
			cout << "Setting .caffemodel file to " << pretrained_binary_proto_file << endl;
		}

		// Text file (.prototxt) defining the network structure
		string feature_extraction_proto_filename = rf.check("feature_extraction_proto_file", Value("imagenet_val_cutfc6.prototxt")).asString().c_str();
		string feature_extraction_proto_file = rf.findFile(feature_extraction_proto_filename);
		if (feature_extraction_proto_file == "")
		{
			feature_extraction_proto_file = contextPath + "/" + feature_extraction_proto_filename;
		}
		cout << "Setting .prototxt file to " << feature_extraction_proto_file << endl;

		// Name of blobs to be extracted
		string extract_features_blob_names = rf.check("extract_features_blob_names", Value("fc6")).asString().c_str();

		// Compute mode and eventually GPU ID to be used
		string compute_mode = rf.check("compute_mode", Value("GPU")).asString();
		int device_id = rf.check("device_id", Value(0)).asInt();

		// Boolean flag for timing or not the feature extraction
		bool timing = rf.check("timing",Value(false)).asBool();

		caffe_extractor = NULL;
		caffe_extractor = new CaffeFeatExtractor<double>(pretrained_binary_proto_file,
				feature_extraction_proto_file,
				extract_features_blob_names,
				compute_mode,
				device_id,
				timing);

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

		Time::turboBoost();

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
		return -1;

	ResourceFinder rf;

	rf.setVerbose(true);

	rf.setDefaultContext("himrep");
	rf.setDefaultConfigFile("caffeCoder.ini");

	rf.configure(argc,argv);

	rf.setDefault("name","caffeCoder");

	CaffeCoderModule mod;

	return mod.runModule(rf);

}

