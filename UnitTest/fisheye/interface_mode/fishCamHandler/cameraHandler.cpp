//
// Created by Orangels on 2020-04-24.
//

#include "cameraHandler.h"
#include <iostream>
#include "string.h"
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;


cameraHandler::~cameraHandler(){

}

cameraHandler::cameraHandler(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        float center_zoom_angle)
        : cameraHandler(input_width, input_height, input_data,
                     output_width, output_height, output_data) {

    unsigned long iResult = SetVideoParams(
            &input_buffer,
            &output_buffer,
            IMV_Defs::E_BGR_24_STD | IMV_Defs::E_OBUF_TOPBOTTOM,
            IMV_Defs::E_VTYPE_PTZ,
            IMV_Defs::E_CPOS_CEILING);

    if (iResult == 0)
    {

        //the library is correctly initialized
        cout << "IMV center Init suc" <<  endl;
    }
    else
    {

        //an error occurred
        cout << "IMV center Init error : " << iResult << endl;

    }
    float pan = 0.0f, tilt = 0.0f, roll = 0.0f, zoom = center_zoom_angle;
    SetPosition(&pan, &tilt, &roll, &zoom);
    SetThreadCount(12);
    Update();
}

cameraHandler::cameraHandler(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        float perimeter_top_angle, float perimeter_bottom_angle)
        : cameraHandler(input_width, input_height, input_data,
                     output_width, output_height, output_data) {

    unsigned long iResult = SetVideoParams(
            &input_buffer,
            &output_buffer,
            IMV_Defs::E_BGR_24_STD | IMV_Defs::E_OBUF_TOPBOTTOM,
            IMV_Defs::E_VTYPE_PERI,
            IMV_Defs::E_CPOS_CEILING);

    if (iResult == 0)

    {

        //the library is correctly initialized
        cout << "IMV perimeter Init suc" << endl;
    }

    else

    {

        //an error occurred
        cout << "IMV perimeter Init error : " << iResult << endl;

    }

    SetTiltLimits(perimeter_top_angle - 90.0f, perimeter_bottom_angle - 90.0f);
    SetThreadCount(12);
    Update();
}

cameraHandler::cameraHandler (
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data
        ) {
    const char *rpl = "B9VVT";
    SetLens(strdup(rpl));
    SetFiltering(IMV_Defs::E_FILTER_BILINEAR);
    input_buffer = (IMV_Buffer) {
            .width = (unsigned long) (input_width),
            .height = (unsigned long) (input_height),
            .frameX = 0,
            .frameY = 0,
            .frameWidth = (unsigned long) (input_width),
            .frameHeight = (unsigned long) (input_height),
            .data = input_data,
    };
    output_buffer = (IMV_Buffer) {
            .width = (unsigned long) (output_width),
            .height = (unsigned long) (output_height),
            .frameX = 0,
            .frameY = 0,
            .frameWidth = (unsigned long) (output_width),
            .frameHeight = (unsigned long) (output_height),
            .data = output_data,
    };
}

int cameraHandler::run(unsigned char *vInput_data) {
    input_buffer.data = vInput_data;
    Update();

    return 0;
}


void cameraHandler::GetInputPointFromOutputPoint(
        int output_x, int output_y, int *input_x, int *input_y) {
    float pan, tilt;
    GetPositionFromOutputVideoPoint(output_x, output_y, &pan, &tilt);
    cout << "pan : " << pan << " tilt : " << tilt <<endl;
    GetInputVideoPointFromPosition(pan, tilt, input_x, input_y);
    cout << "x : " << *input_x << " y : " << *input_y <<endl;
}

void cameraHandler::GetInputPolygonFromOutputPolygon(std::vector<float> &output_, float *input_) {
    int x, y;
    for (unsigned int i = 0; i < output_.size() / 2; i++) {
        GetInputPointFromOutputPoint(int(output_[2 * i]), int(output_[2 * i + 1]), &x, &y);
        input_[2 * i] = float(x), input_[2 * i + 1] = float(y);
    }
}