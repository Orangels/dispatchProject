#include <GL/glew.h>
#include "IMV1.h"
#include "split.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <vector>

//###########
//1. cuda & opengl: https://www.cnblogs.com/csuftzzk/p/cuda_opengl_interoperability.html
//2. OpenGL与CUDA互操作方式总结： https://www.cnblogs.com/csuftzzk/p/cuda_opengl_interoperability.html
//2.2.取tex数据： https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glPixelStore.xhtml
//###########

class CameraView : public IMV_CameraFlatSurfaceInterface {
public:
    CameraView(int input_width, int input_height, unsigned char *input_data,
               int output_width, int output_height, unsigned char *output_data,
               const char *lens_rpl, float center_zoom_angle);

    CameraView(int input_width, int input_height, unsigned char *input_data,
               int output_width, int output_height, unsigned char *output_data,
               const char *lens_rpl,
               float perimeter_top_angle, float perimeter_bottom_angle);

    ~CameraView();

    void Process();

    void GetInputPointFromOutputPoint(int output_x, int output_y, int *input_x, int *input_y);

    void GetInputPolygonFromOutputPolygon(std::vector<float> &output_, float *input_);

    //////////
    void GetOutputPointFromInputPoint(int output_x, int output_y, int *input_x, int *input_y);

    void GetOutputPolygonFromInputPolygon();

    void resetOutput(unsigned char *output_data);

    void resetInput(unsigned char *input_data);

    void getOutput();

    splitProcess *sp;
    IMV_Buffer input_buffer;
    IMV_Buffer output_buffer;
    void *devPtr;
private:
    CameraView(int input_width, int input_height, unsigned char *input_data,
               int output_width, int output_height, unsigned char *output_data,
               const char *lens_rpl);

    void InitGL();


    void UnpackToTexture();

    void RenderScene();

    void PackFromFramebuffer();

    GLuint tex_id;
    GLuint fbo_id;
    GLuint rbo_id;
    GLuint pbo_id[2];

    int num_vertices;
    Vertex2D *vertices;
    Vertex2D *txcoords;

    size_t size;

    //paramers about cuda
    cudaGraphicsResource *resource;

    // show eclipse
    int ntm;
    std::vector<float> tm;
};
