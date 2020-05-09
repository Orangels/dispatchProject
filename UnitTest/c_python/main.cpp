/***************************************
  g++ main.cpp -o main -I/usr/include/python2.7/ -lpython2.7 -lpthread
 ****************************************/

#include "EnginePy.hpp"
#include <time.h>
#include <thread>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "mat2numpy.h"

using namespace std;

int main()
{
    cv::Mat img0 = cv::imread("/home/user/workspace/xxs/DPH_Server_tmp/pycode/test.png");

    int ret = 0;
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        LOG_DEBUG("Py_Initialize error, return\n");
        return -1;
    }

    PyEval_InitThreads();
    int nInit = PyEval_ThreadsInitialized();
    if (nInit)
    {
        LOG_DEBUG("PyEval_SaveThread\n");
        PyEval_SaveThread();
    }

    Engine_api *pPer_0 = new Engine_api("engine_api");

    vector<int> vret0,vret1,vret2,vret3;

    while (true){

    auto thread_func_0 = [&](){
        vret0 = pPer_0->get_result(img0.clone(), "");
        LOG_DEBUG("pPer 0 \n");
    };

//    auto thread_func_1 = [&](){
//        pPer_0->get_result(img0.clone(), "1");
//        LOG_DEBUG("pPer 1 \n");
//    };
////
//    auto thread_func_2 = [&](){
//        pPer_0->get_result(img0.clone(), "2");
//        LOG_DEBUG("pPer 2 \n");
//    };

    thread thread_ctx_0(thread_func_0);
//    thread thread_ctx_1(thread_func_1);
//    thread thread_ctx_2(thread_func_2);

    thread_ctx_0.join();
//    thread_ctx_1.join();
//    thread_ctx_2.join();

//    for (int i = 0; i < vret0.size(); i++){
//        std::cout << vret0[i] << std::endl;
//    }
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_Finalize();
    LOG_DEBUG("main end\n");
    return 0;
}