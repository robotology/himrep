

    #include "SiftGPU_Extractor.h"





    SiftGPU_Extractor::SiftGPU_Extractor()
    {
        SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
        SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
        
        char * pPath;
       pPath = getenv ("SIFTGPU_DIR");
       //printf ("\n\nThe current path is: %s\n\n",pPath);

    /**/
    #ifdef SIFTGPU_DLL_RUNTIME
        string str = pPath;
        #ifdef _WIN32
        str.append("/bin/SIFTGPU.dll");
        #ifdef _DEBUG
            HMODULE  hsiftgpu = LoadLibrary(str.c_str());
        #else
            HMODULE  hsiftgpu = LoadLibrary(str.c_str());
        #endif
    #else
       str.append("/bin/libsiftgpu.so");
       void * hsiftgpu = dlopen(str.c_str(), RTLD_LAZY);
    #endif

    #ifdef REMOTE_SIFTGPU
        ComboSiftGPU* (*pCreateRemoteSiftGPU) (int, char*) = NULL;
        pCreateRemoteSiftGPU = (ComboSiftGPU* (*) (int, char*)) GET_MYPROC(hsiftgpu, "CreateRemoteSiftGPU");
        combo = pCreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
        sift = combo;
        matcher = combo;
    #else
        //SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
        //SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
        pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
        pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
        sift = pCreateNewSiftGPU(1);
        matcher = pCreateNewSiftMatchGPU(4096);
    #endif

    #elif defined(REMOTE_SIFTGPU)
        combo = CreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
        sift = combo;
        matcher = combo;
    #else
        //this will use overloaded new operators
        sift = new SiftGPU;
        matcher = new SiftMatchGPU(4096);
    #endif

       char * argv[] = {(char*)"-fo", (char*)"-1", (char*) "-v",(char*) "1", (char*)"-winpos",(char*)"-maxd", (char*)"1024"};
       int argc = sizeof(argv)/sizeof(char*);


       sift->ParseParam(argc, argv);
        //verbose on sift to remove unwanted printouts (put 1 for data)
       sift->SetVerbose(0);
/////////////////////////////////////////////
   
       if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
           fprintf(stdout,"boh, some error\n");

       matcher->VerifyContextGL();
    }





    bool SiftGPU_Extractor::setDenseGrid(int width, int height, int step, int scale)
    {
        int widthNodes=(int)(width-step)/step;
        int heightNodes=(int)(height-step)/step;

        keypoints_grid.resize(widthNodes*heightNodes);
        int idx=0;
        for(int row=step; row<=height-step; row+=step)
        {
            for(int col=step; col<=width-step; col+=step)
            {
                //int idx=(row/step)*widthNodes + col/step;
                keypoints_grid[idx].x=col;
                keypoints_grid[idx].y=row;
                keypoints_grid[idx].s=scale;
                keypoints_grid[idx].o=0.0f;
                idx++;
            }
        }   
        
        return true;
    }



    bool SiftGPU_Extractor::setDenseGrid(IplImage *img, int step, int scale)
    {
        setDenseGrid(img->width,img->height,step,scale);
    }


    
    
    bool SiftGPU_Extractor::extractSift(IplImage *img,vector<SiftGPU::SiftKeypoint> *keypoints, vector<float> *descriptors, int feature_size)
    {
        if(img->nChannels==1)
            sift->RunSIFT(img->width,img->height,img->imageData,GL_LUMINANCE,GL_UNSIGNED_BYTE ); 
        else if(img->nChannels==3)
            sift->RunSIFT(img->width,img->height,img->imageData,GL_RGB,GL_UNSIGNED_BYTE );
        else
            return false;

        if(feature_size!=128)
            fprintf(stdout,"Error! wrong feature size!\n");

        int feature_num=sift->GetFeatureNum();

        if(feature_num<=0)
          return false;

        if(keypoints!=NULL && descriptors!=NULL)
        {
            keypoints->resize(feature_num);
            descriptors->resize(feature_size*feature_num);
            sift->GetFeatureVector(&(keypoints->at(0)),&(descriptors->at(0)));
        }

       return true;
    }
    
    
    
    bool SiftGPU_Extractor::extractDenseSift(IplImage *img,vector<SiftGPU::SiftKeypoint> *keypoints, vector<float> *descriptors, int feature_size)
    {
        if(keypoints_grid.size()==0)
            return false;
            
        sift->SetKeypointList(keypoints_grid.size(),&keypoints_grid[0],0);
        return this->extractSift(img,keypoints,descriptors,feature_size);
    }

    
    int SiftGPU_Extractor::getFeatureNum()
    {
        return sift->GetFeatureNum();
    }


    bool SiftGPU_Extractor::getFeatureVector(vector<SiftGPU::SiftKeypoint> *keypoints, vector<float> *descriptors, int feature_size)
    {
       if(feature_size!=128)
            fprintf(stdout,"Error! wrong feature size!\n");
    
       int feature_num=sift->GetFeatureNum();
       
       keypoints->resize(feature_num);
       descriptors->resize(feature_size*feature_num);
       sift->GetFeatureVector(&(keypoints->at(0)),&(descriptors->at(0)));
       return true;
    }

