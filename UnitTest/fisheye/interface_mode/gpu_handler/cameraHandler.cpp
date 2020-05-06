//
// Created by Orangels on 2020-04-24.
//

#include "cameraHandler.h"
#include <iostream>
#include "string.h"
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;

void cameraHandler::InitGL() {
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0, 3,
                 input_buffer.width, input_buffer.height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &fbo_id);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
    glGenRenderbuffers(1, &rbo_id);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo_id);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8,
                          output_buffer.width, output_buffer.height);
    glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo_id);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenBuffers(2, pbo_id);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id[0]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 3 * input_buffer.width * input_buffer.height,
                 NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id[1]);

    // Allocate data for the buffer
    glBufferData(GL_PIXEL_PACK_BUFFER,
                 3 * output_buffer.width * output_buffer.height,
                 NULL, GL_DYNAMIC_COPY);
    cudaGraphicsGLRegisterBuffer(&resource, pbo_id[1],
                                 cudaGraphicsMapFlagsNone);//#9.6

}

cameraHandler::~cameraHandler(){
    cudaGraphicsUnregisterResource(resource);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glDeleteBuffers(2, pbo_id);

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glDeleteRenderbuffers(1, &rbo_id);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo_id);

    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &tex_id);

    ipt_pixels_dyn_ls = NULL;
}

cameraHandler::cameraHandler(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        float center_zoom_angle, int vInferMode)
        : cameraHandler(input_width, input_height, input_data,
                     output_width, output_height, output_data, vInferMode) {

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
//    SetThreadCount(12);
    Update();
    if (inferMode == 0){
        GetFlatSurfaceModel(0, &num_vertices, &vertices, &txcoords);
    }
}

cameraHandler::cameraHandler(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        float perimeter_top_angle, float perimeter_bottom_angle, int vInferMode)
        : cameraHandler(input_width, input_height, input_data,
                     output_width, output_height, output_data, vInferMode) {

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
//    SetThreadCount(12);
    Update();
    if (inferMode == 0){
        GetFlatSurfaceModel(0, &num_vertices, &vertices, &txcoords);
    }
}

cameraHandler::cameraHandler(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        int vInferMode
        ) : inferMode(vInferMode) {
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
    if (inferMode==0){
        InitGL();
    }
}

int cameraHandler::run(unsigned char *vInput_data) {
    if (inferMode==1){
        input_buffer.data = vInput_data;
        Update();
    } else if (inferMode==0){
        Process();
        cudaMemcpy(output_buffer.data, ipt_pixels_dyn_ls, output_buffer.width * output_buffer.height * sizeof(uchar) * 3, cudaMemcpyDeviceToHost);
    }

    return 0;
}

void cameraHandler::Process() {
    auto start = std::chrono::steady_clock::now();
    UnpackToTexture();

    auto stop = std::chrono::steady_clock::now();
    auto timing = std::chrono::duration_cast < std::chrono::duration < double >> (stop - start);
//    tm.push_back(1000.0 * (end - beg) / CLOCKS_PER_SEC);
    std::cout << "Unpacking: " << timing.count() * 1000.0  << "ms ";
//              << 1000.0 * (end - beg) / CLOCKS_PER_SEC
//              << "ms ";
//    std::cout.flush();

    start = std::chrono::steady_clock::now();
    RenderScene();
    stop = std::chrono::steady_clock::now();
    timing = std::chrono::duration_cast < std::chrono::duration < double >> (stop - start);
//    tm.push_back(1000.0 * (end - beg) / CLOCKS_PER_SEC);
    std::cout << "Rendering: " << timing.count() * 1000.0 <<  "ms ";
//              << 1000.0 * (end - beg) / CLOCKS_PER_SEC
//              << "ms ";
//    std::cout.flush();

    start = std::chrono::steady_clock::now();
    PackFromFramebuffer();

    stop = std::chrono::steady_clock::now();
    timing = std::chrono::duration_cast < std::chrono::duration < double >> (stop - start);
//    tm.push_back(1000.0 * (end - beg) / CLOCKS_PER_SEC);
//    ntm++;
    std::cout << "Packing: " << timing.count() * 1000.0 << "ms\n";
//              << 1000.0 * (end - beg) / CLOCKS_PER_SEC
//              << "ms\n";
//    std::cout.flush();

//    GetOutputPolygonFromInputPolygon();
}

void cameraHandler::UnpackToTexture() {
    glBindTexture(GL_TEXTURE_2D, tex_id);
    GLubyte *ptr = (GLubyte *) glMapBuffer(
            GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        memcpy(ptr, input_buffer.data,
               3 * input_buffer.width * input_buffer.height);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    input_buffer.width, input_buffer.height,
                    GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

void cameraHandler::RenderScene() {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);

    glViewport(0, 0, output_buffer.width, output_buffer.height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, output_buffer.width, 0, output_buffer.height, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClear(GL_COLOR_BUFFER_BIT);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

//    printf("yyyyy\n" );
    glVertexPointer(2, GL_FLOAT, 0, vertices);
    glTexCoordPointer(2, GL_FLOAT, 0, txcoords);
    glDrawArrays(GL_TRIANGLES, 0, num_vertices);
//    printf("xxxx\n" );

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

void cameraHandler::PackFromFramebuffer() {
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, output_buffer.width, output_buffer.height,
                 GL_RGB, GL_UNSIGNED_BYTE, NULL);
    cudaGraphicsMapResources(1, &resource, NULL);

//    std::cout.flush();
//    cudaGraphicsResourceGetMappedPointer((void **) &(sp->ipt_pixels_dyn), &size, resource);

    cudaGraphicsResourceGetMappedPointer((void **) &(ipt_pixels_dyn_ls), &size, resource);

    //cudaGraphicsUnmapResources(1, &resource, NULL);//好像可以去掉
}