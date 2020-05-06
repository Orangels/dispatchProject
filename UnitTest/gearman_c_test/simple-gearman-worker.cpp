/*
 * File:
 *    simple-gearman-worker.cpp
 * Auth:
 * Hank(hongkuiyan@yeah.net)
 * Desc:
 * Example code to show how to receive a string to a function called "test_function"
 * Cypt:
 * Gearman server and library Version 1.1.12
 * Compile:
 * g++ -o simple-gearman-worker simple-gearman-worker.cpp ./base64/base64.cpp -I/usr/local/include -I/usr/include/opencv4/  -L/usr/local/lib -lgearman -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -std=c++11
 * Usage:
 * ./simple-gearman-worker
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
# include <vector>
#include <libgearman/gearman.h>
#include <opencv2/highgui.hpp>
#include "base64/base64.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

using namespace std;
using namespace cv;

#define private public

static void *test_function(gearman_job_st *job,
                           void* context,
                           size_t* result_size,
                           gearman_return_t* ret_ptr);


string mode = "0";

int main(int argc, char* argv[])
{
    if (argc < 2){
        cout << "Usage mode 0 to bytes and mode 1 to base64" << endl;
        return -1;
    }

    mode = string(argv[1]);

    gearman_return_t gearRet;
    gearman_worker_st* gearWorker;

    char* gearSvrHost=(char*)"127.0.0.1", *gearSvrPort=(char*)"4730";
    char* gearContext = new char(50);

    gearWorker = gearman_worker_create(NULL);
    if (gearWorker == NULL)
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
        return EXIT_FAILURE;
    }

    gearRet = gearman_worker_add_server(gearWorker, gearSvrHost, atoi(gearSvrPort));
    if (gearman_failed(gearRet))
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
        return EXIT_FAILURE;
    }

    gearRet = gearman_worker_add_function(gearWorker,
                                          "DPH_TRACKER_SERVER",
                                          0,
                                          test_function,
                                          gearContext);



    if (gearman_failed(gearRet))
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
        return EXIT_FAILURE;
    }

    cout << "waiting for job ... " << endl;

    while (1)
    {
        gearRet = gearman_worker_work(gearWorker);
        if (gearman_failed(gearRet))
        {
            cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
            break;
        }
    }

    delete gearContext;
    gearman_worker_free(gearWorker);

    return EXIT_SUCCESS;

}

static void *test_function(gearman_job_st *job,
                           void *context,
                           size_t *result_size,
                           gearman_return_t *ret_ptr)
{
    (void)context;

    const char *workload;
    workload= (const char *)gearman_job_workload(job);
    *result_size= gearman_job_workload_size(job);

    uint64_t count= 0;
//    if (workload != NULL)
//    {
//        if (workload[0] != ' ' && workload[0] != '\t' && workload[0] != '\n')
//            count++;
//
//        for (size_t x= 0; x < *result_size; x++)
//        {
//            if (workload[x] != ' ' && workload[x] != '\t' && workload[x] != '\n')
//                continue;
//
//            count++;
//
//            while (workload[x] == ' ' || workload[x] == '\t' || workload[x] == '\n')
//            {
//                x++;
//                if (x == *result_size)
//                {
//                    count--;
//                    break;
//                }
//            }
//        }
//    }


    std::string result= "asdasd";
    std::cerr << "Job= " << gearman_job_handle(job) << " Workload= " << workload << endl;
//    std::cerr << workload << std::endl;
//    std::cerr.write(workload, *result_size);

//    std::cerr << " Result=" << workload << std::endl;
    std::cout << *result_size << endl;


    if (mode == "0"){
        cout << "mode 0 bytes" << endl;

        string jsonStr = workload;
//        cout << workload << endl;
        rapidjson::Document dom;
        if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
            if (dom.HasMember("imgs") && dom["imgs"].IsString()) {
                cout << "write img" << endl;
                auto img_str = dom["imgs"].GetString();
                auto img_len = dom["imgs"].GetStringLength();
                cout << "str len : " << dom["imgs"].GetStringLength() << endl;
                Mat image;
                std::vector<uchar> decode;
                for (int i = 0; i < img_len; ++i) {
//                    cout << i << endl;
                    decode.push_back(img_str[i]);
                }
                cout << "333" << endl;
                image = imdecode(decode, IMREAD_COLOR);//图像解码
                imwrite("./imgs/worker_test.jpg", image);

            }
            if (dom.HasMember("bbox") && dom["bbox"].IsArray()) {
                const rapidjson::Value& arr = dom["bbox"];

                printf("bbox: ");
                for (int i = 0; i < arr.Size(); ++i) {
                    printf("%d, ", arr[i].GetInt());
                }
                printf("\n");
            }

        } else{
            printf("fail to parse json str\n");
        }

//                Mat image;
//                std::vector<uchar> decode;
//                for (int i = 0; i < *result_size; ++i) {
//                    decode.push_back(workload[i]);
//                }
//                image = imdecode(decode, IMREAD_COLOR);//图像解码
//                imwrite("./imgs/worker_test.jpg", image);

    } else if (mode == "1"){
        cout << "mode 1 base64" << endl;

        string jsonStr = workload;
        rapidjson::Document dom;
        if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
            if (dom.HasMember("imgs") && dom["imgs"].IsString()) {
                auto img_str = dom["imgs"].GetString();
                std::string decoded = base64_decode(img_str);

                Mat image;
                std::vector<uchar> decode;
                for (int i = 0; i < decoded.length(); ++i) {
                    decode.push_back(decoded[i]);
                }
                image = imdecode(decode, IMREAD_COLOR);//图像解码
                imwrite("./imgs/worker_b64_test.jpg", image);

            }
            if (dom.HasMember("bbox") && dom["bbox"].IsArray()) {
                const rapidjson::Value& arr = dom["bbox"];

                printf("bbox: ");
                for (int i = 0; i < arr.Size(); ++i) {
                    printf("%d, ", arr[i].GetInt());
                }
            }

        } else{
            printf("fail to parse json str\n");
        }
    } else if (mode == "2"){
        cout << "mode 2 json" << endl;
        cout << "params ori : " << workload << endl;
        string jsonStr = workload;
        jsonStr= jsonStr.substr(0, static_cast<int>(*result_size));
        cout << "params json : " << jsonStr << endl;
        rapidjson::Document dom;
        if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
            if (dom.HasMember("persons") && dom["persons"].IsArray()){
                const rapidjson::Value& arr = dom["persons"];

                for (int i = 0; i < arr.Size(); ++i) {
                    if (arr[i].HasMember("id") && arr[i]["id"].IsInt()) {
                        int person_id = arr[i]["id"].GetInt();
                        cout << "person id : " << person_id << endl;
                    }
                }
            }
        }else{
            printf("fail to parse json str\n");
        }
    }


    *result_size= result.size();

    *ret_ptr= GEARMAN_SUCCESS;
    return strdup(result.c_str());
}
