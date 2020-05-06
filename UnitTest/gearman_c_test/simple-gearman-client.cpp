/*
 * File:
 *    simple-gearman-client.cpp
 * Auth:
 * Hank(hongkuiyan@yeah.net)
 * Desc:
 * Example code to show how to send a string to a function called "test_function" .
 * Cypt:
 * Gearman server and library Version 1.1.12
 * Compile:
 * g++ -o simple-gearman-client simple-gearman-client.cpp ./base64/base64.cpp -I/usr/local/include -I/usr/include/opencv4/  -L/usr/local/lib -lgearman -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -std=c++11
 * Usage:
 * ./simple-gearman-client
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgearman/gearman.h>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "base64/base64.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"


using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
    if (argc < 2){
        cout << "Usage mode 0 to bytes and mode 1 to base64" << endl;
        return -1;
    }

    gearman_return_t gearRet;
    gearman_client_st* gearClient;

//    char* gearSvrHost=(char*)"127.0.0.1", *gearSvrPort=(char*)"4730";
    char* gearSvrHost=(char*)"192.168.88.221", *gearSvrPort=(char*)"4730";

    gearClient = gearman_client_create(NULL);
    gearman_client_set_options(gearClient, GEARMAN_CLIENT_FREE_TASKS);
    gearman_client_set_timeout(gearClient, 15000);

    gearRet = gearman_client_add_server(gearClient, gearSvrHost, atoi(gearSvrPort));
    if (gearman_failed(gearRet))
    {
        return EXIT_FAILURE;
    }

#if 1

    vector<uchar> data_encode;
    Mat img = imread("./imgs/ls.jpg");
    imencode(".jpg", img, data_encode);
    int arr_length = data_encode.size();
    char* encodeImg = new char[arr_length];
    cout << "img length : " << arr_length << endl;
    for (int i = 0; i < data_encode.size(); ++i) {
        encodeImg[i] = data_encode[i];
    }

    if (string(argv[1]) == "0"){
        cout << "mode 0 bytes" << endl;

        rapidjson::StringBuffer buf;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buf);

        writer.StartObject();

        writer.Key("imgs");writer.String(encodeImg, arr_length);
        writer.Key("bbox");

        writer.StartArray();
        writer.Int(274);
        writer.Int(212);
        writer.Int(435);
        writer.Int(435);
        writer.EndArray();

        writer.EndObject();

        const char* json_content = buf.GetString();
        string json_cs = json_content;

        gearRet = gearman_client_do_background(gearClient,
                                               "DHP_face",
                                               NULL,
                                               json_cs.c_str(),
                                               json_cs.length(),
                                               NULL);
//        gearRet = gearman_client_do_background(gearClient,
//                                               "dph",
//                                               NULL,
//                                               encodeImg,
//                                               arr_length,
//                                               NULL);


    }
    else if (string(argv[1]) == "1"){
        cout << "mode 1 base64" << endl;
        std::string encoded = base64_encode(reinterpret_cast<const unsigned char*>(encodeImg), arr_length);

        rapidjson::StringBuffer buf;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buf);

        writer.StartObject();

        writer.Key("imgs");writer.String(encoded.c_str());
        writer.Key("bbox");

        writer.StartArray();
        for (int i = 0; i < 1; ++i) {
            writer.StartArray();
            writer.Int(274 + i*10);
            writer.Int(212 + i*10);
            writer.Int(435 + i*10);
            writer.Int(435 + i*10);
            writer.EndArray();
        }
        writer.EndArray();

        writer.EndObject();

        const char* json_content = buf.GetString();
        string json_cs = json_content;

        gearRet = gearman_client_do_background(gearClient,
                                               "DHP_face",
                                               NULL,
                                               json_cs.c_str(),
                                               json_cs.length(),
                                               NULL);
    } else{
        cout << "Unknow mode " << argv[1] << endl;
    }
    if (gearRet == GEARMAN_SUCCESS)
    {
        fprintf(stdout, "Work success!\n");
    }
    else if (gearRet == GEARMAN_WORK_FAIL)
    {
        fprintf(stderr, "Work failed\n");
    }
    else if (gearRet == GEARMAN_TIMEOUT)
    {
        fprintf(stderr, "Work timeout\n");
    }
    else
    {
        fprintf(stderr, "%d,%s\n", gearman_client_errno(gearClient), gearman_client_error(gearClient));
    }


#endif


#if 0
    gearman_argument_t gearValue = gearman_argument_make(0, 0,
        "Reverse Me", strlen("Reverse Me"));
    gearman_task_st* gearTask = gearman_execute(gearClient,
        "reverse", strlen("reverse"),
        NULL, 0,
        NULL,
        &gearValue, 0);

    /* If gearman_execute() can return NULL on error */
    if (gearTask == NULL)
    {
        fprintf(stderr, "Error: %s\n", gearman_client_error(gearClient));
        gearman_client_free(gearClient);
        return EXIT_FAILURE;
    }

    /* Make sure the task was run successfully */
    if (gearman_success(gearman_task_return(gearTask)))
    {
        /* Make use of value */
        gearman_result_st *gearResult= gearman_task_result(gearTask);
        printf("%.*s\n", (int)gearman_result_size(gearResult), gearman_result_value(gearResult));
    }
#endif

    gearman_client_free(gearClient);

    return EXIT_SUCCESS;
}