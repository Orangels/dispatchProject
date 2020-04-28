//
// Created by Orangels on 2020-04-24.
//

#ifndef FISHEYE_CAMERAHANDLER_H
#define FISHEYE_CAMERAHANDLER_H

#include "IMV1.h"

class cameraHandler : public IMV_CameraInterface {
//class cameraHandler : public IMV_CameraFlatSurfaceInterface {
public:
    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
               int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
               float vCenter_zoom_angle);

    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
               float vPerimeter_top_angle, float vPerimeter_bottom_angle);

    ~cameraHandler(){
        
    }

    int run(unsigned char *vInput_data);

    IMV_Buffer input_buffer;
    IMV_Buffer output_buffer;

private:
    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data);
};


#endif //FISHEYE_CAMERAHANDLER_H
