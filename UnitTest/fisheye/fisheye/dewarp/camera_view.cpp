#ifdef __JETBRAINS_IDE__
#define DEFINE_bool
#define DEFINE_double
//how it works: https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/46101718#46101718
#endif

#include "camera_view.h"
#include <ctime>
#include <iostream>
#include <string.h>

//#include <gflags/gflags.h>
#include <GL/glew.h>
#include "IMV1.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <chrono>

//DEFINE_bool(use_pbo, true, "whether to use PBO for unpacking/packing");

unsigned long _tmp_GetTickCount() {
    // https://blog.csdn.net/guang11cheng/article/details/6865992
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

CameraView::CameraView(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        const char *lens_rpl, float center_zoom_angle)
        : CameraView(input_width, input_height, input_data,
                     output_width, output_height, output_data, lens_rpl) {
    SetVideoParams(&input_buffer, &output_buffer,
                   IMV_Defs::E_BGR_24_STD | IMV_Defs::E_OBUF_TOPBOTTOM,
                   IMV_Defs::E_VTYPE_PTZ,
                   IMV_Defs::E_CPOS_CEILING);

    float pan = 0.0f, tilt = 0.0f, roll = 0.0f, zoom = center_zoom_angle;
    SetPosition(&pan, &tilt, &roll, &zoom);

    Update();
    GetFlatSurfaceModel(0, &num_vertices, &vertices, &txcoords);
}

CameraView::CameraView(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        const char *lens_rpl,
        float perimeter_top_angle, float perimeter_bottom_angle)
        : CameraView(input_width, input_height, input_data,
                     output_width, output_height, output_data, lens_rpl) {

    SetVideoParams(&input_buffer, &output_buffer,
                   IMV_Defs::E_BGR_24_STD | IMV_Defs::E_OBUF_TOPBOTTOM,
                   IMV_Defs::E_VTYPE_PERI,
                   IMV_Defs::E_CPOS_CEILING);

    SetTiltLimits(perimeter_top_angle - 90.0f, perimeter_bottom_angle - 90.0f);

    Update();
    GetFlatSurfaceModel(0, &num_vertices, &vertices, &txcoords);
}

CameraView::CameraView(
        int input_width, int input_height, unsigned char *input_data,
        int output_width, int output_height, unsigned char *output_data,
        const char *lens_rpl) {
    SetLens(strdup(lens_rpl));
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

    InitGL();
}

void CameraView::resetOutput(unsigned char *output_data) {
    output_buffer.data = output_data;
}

void CameraView::resetInput(unsigned char *input_data) {
    input_buffer.data = input_data;
}

void CameraView::InitGL() {
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

CameraView::~CameraView() {
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

    float u = 0, r = 0, p = 0;
    for (auto i = 0; i < tm.size(); i += 3) {
        u += tm[i];
        r += tm[i + 1];
        p += tm[i + 2];
    }
    std::cout << "Unpacking: " << u / ntm << "ms Rendering: " << r / ntm << "ms Packing: " << p / ntm << "ms\n";
}

void CameraView::Process() {
//    std::clock_t beg = std::clock();
//    unsigned long tbeg = _tmp_GetTickCount();
    auto start = std::chrono::steady_clock::now();
    UnpackToTexture();
//    std::clock_t end = std::clock();
//    unsigned long tend = _tmp_GetTickCount();
    auto stop = std::chrono::steady_clock::now();
    auto timing = std::chrono::duration_cast < std::chrono::duration < double >> (stop - start);
//    tm.push_back(1000.0 * (end - beg) / CLOCKS_PER_SEC);
    std::cout << "Unpacking: " << timing.count() * 1000.0  << "ms ";
//              << 1000.0 * (end - beg) / CLOCKS_PER_SEC
//              << "ms ";
//    std::cout.flush();

//    beg = std::clock();
//    tbeg = _tmp_GetTickCount();
    start = std::chrono::steady_clock::now();
    RenderScene();
//    end = std::clock();
//    tend = _tmp_GetTickCount();
    stop = std::chrono::steady_clock::now();
    timing = std::chrono::duration_cast < std::chrono::duration < double >> (stop - start);
//    tm.push_back(1000.0 * (end - beg) / CLOCKS_PER_SEC);
    std::cout << "Rendering: " << timing.count() * 1000.0 <<  "ms ";
//              << 1000.0 * (end - beg) / CLOCKS_PER_SEC
//              << "ms ";
//    std::cout.flush();

//    beg = std::clock();
//    tbeg = _tmp_GetTickCount();
    start = std::chrono::steady_clock::now();
    PackFromFramebuffer();
//    end = std::clock();
//    tend = _tmp_GetTickCount();
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

void CameraView::UnpackToTexture() {
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

void CameraView::RenderScene() {
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

void CameraView::PackFromFramebuffer() {
//    printf("11111111\n" );
    glReadBuffer(GL_COLOR_ATTACHMENT0);
//    printf("222222" );
    glReadPixels(0, 0, output_buffer.width, output_buffer.height,
                 GL_RGB, GL_UNSIGNED_BYTE, NULL);
//    printf("333333" );
    cudaGraphicsMapResources(1, &resource, NULL);
//    printf("444444" );

//    std::cout.flush();
    cudaGraphicsResourceGetMappedPointer((void **) &(sp->ipt_pixels_dyn), &size, resource);
    /**/
//    std::cout << "____>run " << std::endl;
//    sp->run();
//    std::cout << "____>get " << std::endl;
//    sp->get_2p_Img(output_buffer.data);#######
    /**/

    //cudaGraphicsUnmapResources(1, &resource, NULL);//好像可以去掉
}

void CameraView::getOutput() {
    sp->get_2p_Img(output_buffer.data);
}

void CameraView::GetInputPointFromOutputPoint(
        int output_x, int output_y, int *input_x, int *input_y) {
    float pan, tilt;
    GetPositionFromOutputVideoPoint(output_x, output_y, &pan, &tilt);
    GetInputVideoPointFromPosition(pan, tilt, input_x, input_y);
}

void CameraView::GetInputPolygonFromOutputPolygon(std::vector<float> &output_, float *input_) {
    int x, y;
    for (unsigned int i = 0; i < output_.size() / 2; i++) {
        GetInputPointFromOutputPoint(int(output_[2 * i]), int(output_[2 * i + 1]), &x, &y);
        input_[2 * i] = float(x), input_[2 * i + 1] = float(y);
    }
}

///////////////////////////////////////////////////
void CameraView::GetOutputPointFromInputPoint(
        int output_x, int output_y, int *input_x, int *input_y) {
    float pan, tilt;
    unsigned long viewIndex = 1;
    GetPositionFromInputVideoPoint(output_x, output_y, &pan, &tilt);
    GetOutputVideoPointFromPosition(pan, tilt, input_x, input_y, 1);
}

void CameraView::GetOutputPolygonFromInputPolygon() {//std::vector<float> &output_, float *input_
    std::vector<float> output_ = {198, 677, 332, 336, 761, 420, 591, 940};
    // 2p: 1045,497; 293,165;960,256;549,503
    //4 1 2 3
    float input_[8];
    int x, y;
    std::cout << "GetOutputPolygonFromInputPolygon" << std::endl;
    for (unsigned int i = 0; i < output_.size() / 2; i++) {
        GetInputPointFromOutputPoint(int(output_[2 * i] * 2.5), int(output_[2 * i + 1] * 2.5), &x, &y);
        input_[2 * i] = float(x), input_[2 * i + 1] = float(y);
        std::cout << x * 0.4 << ',' << y * 0.4 << ':' << output_[i * 2] << ',' << output_[i * 2 + 1] << "  |  ";
    }
    std::cout << std::endl;
}
