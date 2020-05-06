#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include<iostream>
using namespace std;
using namespace cv;

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080

#define MUXER_BATCH_TIMEOUT_USEC 4000000

gint frame_number = 0;
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
                                  "Roadsign"
};

static gboolean
link_elements_with_filter (GstElement *element1, GstElement *element2)
{
    gboolean link_ok;
    GstCaps *caps;

    caps = gst_caps_new_simple ("video/x-raw",
                                "format", G_TYPE_STRING, "BGRx",
                                "width", G_TYPE_INT, MUXER_OUTPUT_WIDTH,
                                "height", G_TYPE_INT, MUXER_OUTPUT_HEIGHT,
                                NULL);

    link_ok = gst_element_link_filtered (element1, element2, caps);
    gst_caps_unref (caps);

    if (!link_ok) {
        g_warning ("Failed to link element1 and element2!");
    }

    return link_ok;
}




static void createAlphaMat(Mat &mat)
{
    CV_Assert(mat.channels() == 4);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            Vec4b& bgra = mat.at<Vec4b>(i, j);
            bgra[0] = UCHAR_MAX; // Blue
            bgra[1] = saturate_cast<uchar>((float (mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX); // Green
            bgra[2] = saturate_cast<uchar>((float (mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX); // Red
            bgra[3] = saturate_cast<uchar>(0.5 * (bgra[1] + bgra[2])); // Alpha
        }
    }
}


int write_frame(GstBuffer *buf)
{
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_user_meta = NULL;
    // Get original raw data
    GstMapInfo in_map_info;
    char* src_data = NULL;
    if (!gst_buffer_map (buf, &in_map_info, GST_MAP_READ)) {
        g_print ("Error: Failed to map gst buffer\n");
        gst_buffer_unmap (buf, &in_map_info);
        return GST_PAD_PROBE_OK;
    }
    NvBufSurface *surface = (NvBufSurface *)in_map_info.data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
    l_frame = batch_meta->frame_meta_list;
    if (l_frame) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        /* Validate user meta */
        src_data = (char*) malloc(surface->surfaceList[frame_meta->batch_id].dataSize);
        if(src_data == NULL) {
            g_print("Error: failed to malloc src_data \n");
        }
//#ifdef PLATFORM_TEGRA
//
//        NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
//        NvBufSurfacePlaneParams *pParams = &surface->surfaceList[frame_meta->batch_id].planeParams;
//        unsigned int offset = 0;
//        for(unsigned int num_planes=0; num_planes < pParams->num_planes; num_planes++){
//            if(num_planes>0)
//                offset += pParams->height[num_planes-1]*(pParams->bytesPerPix[num_planes-1]*pParams->width[num_planes-1]);
//            for (unsigned int h = 0; h < pParams->height[num_planes]; h++) {
//              memcpy((void *)(src_data+offset+h*pParams->bytesPerPix[num_planes]*pParams->width[num_planes]),
//                    (void *)((char *)surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[num_planes]+h*pParams->pitch[num_planes]),
//                    pParams->bytesPerPix[num_planes]*pParams->width[num_planes]
//                    );
//            }
//        }
//        NvBufSurfaceSyncForDevice (surface, -1, -1);
//        NvBufSurfaceUnMap (surface, -1, -1);
//#else
//        cudaMemcpy((void*)src_data,
//                   (void*)surface->surfaceList[frame_meta->batch_id].dataPtr,
//                   surface->surfaceList[frame_meta->batch_id].dataSize,
//                   cudaMemcpyDeviceToHost);
//
////        cudaMemcpy((void*)src_data,
////                   (void*)surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0],
////                   surface->surfaceList[frame_meta->batch_id].dataSize,
////                   cudaMemcpyDeviceToHost);
//#endif

// ls change mode

#ifdef PLATFORM_TEGRA

        cudaMemcpy((void*)src_data,
           (void*)surface->surfaceList[frame_meta->batch_id].dataPtr,
           surface->surfaceList[frame_meta->batch_id].dataSize,
           cudaMemcpyDeviceToHost);
#else

        NvBufSurfaceMap (surface, -1, -1, NVBUF_MAP_READ);
        NvBufSurfacePlaneParams *pParams = &surface->surfaceList[frame_meta->batch_id].planeParams;
        unsigned int offset = 0;
        for(unsigned int num_planes=0; num_planes < pParams->num_planes; num_planes++){
            if(num_planes>0)
                offset += pParams->height[num_planes-1]*(pParams->bytesPerPix[num_planes-1]*pParams->width[num_planes-1]);
            for (unsigned int h = 0; h < pParams->height[num_planes]; h++) {
                memcpy((void *)(src_data+offset+h*pParams->bytesPerPix[num_planes]*pParams->width[num_planes]),
                       (void *)((char *)surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[num_planes]+h*pParams->pitch[num_planes]),
                       pParams->bytesPerPix[num_planes]*pParams->width[num_planes]
                );
            }
        }
        NvBufSurfaceSyncForDevice (surface, -1, -1);
        NvBufSurfaceUnMap (surface, -1, -1);
#endif



        gint frame_width = (gint)surface->surfaceList[frame_meta->batch_id].width;
        gint frame_height = (gint)surface->surfaceList[frame_meta->batch_id].height;
        gint frame_step = surface->surfaceList[frame_meta->batch_id].pitch;

//        void *frame_data = surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0];
//        cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC4, frame_data, frame_step);

        cv::Mat frame = cv::Mat(frame_height, frame_width, CV_8UC4, src_data, frame_step);
        g_print("%d\n",frame.channels());
        g_print("%d\n",frame.rows);
        g_print("%d\n",frame.cols);

        string img_path;
        img_path = "./imgs/" + to_string(frame_number) + ".jpg";

        cv::Mat out_mat = cv::Mat (cv::Size(frame_width, frame_height), CV_8UC3);
        cv::cvtColor(frame, out_mat, CV_RGBA2BGR);
        cv::imwrite(img_path, out_mat);
        if(src_data != NULL) {
            free(src_data);
            src_data = NULL;
        }
    }
    gst_buffer_unmap (buf, &in_map_info);
}


static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data){
//    if(++frame_number > 1) return GST_PAD_PROBE_OK;

    frame_number++;

    GstBuffer *buf = (GstBuffer *) info->data;

    write_frame(buf);

//    GstMapInfo in_map_info;
//    NvBufSurface *surface = NULL;
//    NvDsBatchMeta *batch_meta = NULL;
//    NvDsMetaList *l_frame = NULL;
//    NvDsFrameMeta *frame_meta = NULL;
//
//    memset (&in_map_info, 0, sizeof (in_map_info));
//
//    if (gst_buffer_map (buf, &in_map_info, GST_MAP_READWRITE)){
//        surface = (NvBufSurface *) in_map_info.data;
//
//        g_print("%d \n" , NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE));
//        NvBufSurfaceSyncForCpu(surface, -1, -1);
//
//        batch_meta = gst_buffer_get_nvds_batch_meta(buf);
//
//        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next){
//            frame_meta = (NvDsFrameMeta *)(l_frame->data);
//            gint frame_width = (gint)surface->surfaceList[frame_meta->batch_id].width;
//            gint frame_height = (gint)surface->surfaceList[frame_meta->batch_id].height;
//            void *frame_data = surface->surfaceList[frame_meta->batch_id].mappedAddr.addr[0];
//            size_t frame_step = surface->surfaceList[frame_meta->batch_id].pitch;
//
////            cv::Mat frame = cv::Mat(frame_height*1.5, frame_width, CV_8UC1, frame_data, frame_step);
//            cv::Mat frame = cv::Mat(frame_height,frame_width, CV_8UC1, frame_data, frame_step);
//            //cv::Mat dst;
//            //cv::resize(frame, dst, cv::Size(frame_width, frame_height));
//            //cv::Mat out_mat = cv::Mat (cv::Size(frame_width, frame_height), CV_8UC3);
//            //cv::cvtColor(frame, out_mat, CV_RGBA2RGB);
//
//            cout << "********" << endl;
//            g_print("%d\n",frame.channels());
//            g_print("%d\n",frame.rows);
//            g_print("%d\n",frame.cols);
//            cout << "!!!!!!!!" << endl;
//            cout << frame_width << endl;
//            cout << frame_height << endl;
//            cout << "********" << endl;
//
//            string img_path;
//            img_path = "./imgs/" + to_string(frame_number) + ".jpg";
//            cv::Mat out_mat = cv::Mat (cv::Size(frame_width, frame_height), CV_8UC3);
//            cv::cvtColor(frame, out_mat, CV_YUV2RGB_NV21);
////            createAlphaMat(frame);
//            /*createAlphaMat(frame);
//            vector<int> compression_params;
//            compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//            compression_params.push_back(9);*/
//            cv::imwrite(img_path, frame);
////            cv::imwrite("test.jpg", out_mat);
//        }
//
//        NvBufSurfaceUnMap(surface, -1, -1);
//    }
//
//    gst_buffer_unmap (buf, &in_map_info);

    return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print ("End of stream\n");
            g_main_loop_quit (loop);
            break;
        case GST_MESSAGE_ERROR:{
            gchar *debug;
            GError *error;
            gst_message_parse_error (msg, &error, &debug);
            g_printerr ("ERROR from element %s: %s\n",
                        GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr ("Error details: %s\n", debug);
            g_free (debug);
            g_error_free (error);
            g_main_loop_quit (loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int
main (int argc, char *argv[])
{
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
            *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
            *nvosd = NULL;
#ifdef PLATFORM_TEGRA
    GstElement *transform = NULL;
#endif
    GstBus *bus = NULL;
    guint bus_watch_id;
    GstPad *osd_sink_pad = NULL;

    /* Check input arguments */
    if (argc != 2) {
        g_printerr ("Usage: %s <H264 filename>\n", argv[0]);
        return -1;
    }

    /* Standard GStreamer initialization */
    gst_init (&argc, &argv);
    loop = g_main_loop_new (NULL, FALSE);

    /* Create gstreamer elements */
    /* Create Pipeline element that will form a connection of other elements */
    pipeline = gst_pipeline_new ("dstest1-pipeline");

    /* Source element for reading from the file */
    source = gst_element_factory_make ("filesrc", "file-source");

    /* Since the data format in the input file is elementary h264 stream,
     * we need a h264parser */
    h264parser = gst_element_factory_make ("h264parse", "h264-parser");

    /* Use nvdec_h264 for hardware accelerated decode on GPU */
    decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

    /* Create nvstreammux instance to form batches from one or more sources. */
    streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

    if (!pipeline || !streammux) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

    /* Use nvinfer to run inferencing on decoder's output,
     * behaviour of inferencing is set through config file */
//    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

    /* Use convertor to convert from NV12 to RGBA as required by nvosd */
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");


    /* Create OSD to draw on the converted RGBA buffer */
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

    /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
    transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
#endif
    //sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    sink = gst_element_factory_make ("fakesink", "nvvideo-renderer");

    if (!source || !h264parser || !decoder
        || !nvvidconv || !nvosd || !sink) {
        g_printerr ("One element could not be created. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

    /* we set the input filename to the source element */
    g_object_set (G_OBJECT (source), "location", argv[1], NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT, "batch-size", 1,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

//    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
//                  MUXER_OUTPUT_HEIGHT, "batch-size", 1, "format", "RGBA",
//                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);


    /* Set all the necessary properties of the nvinfer element,
     * the necessary ones are : */
//    g_object_set (G_OBJECT (pgie),
//                  "config-file-path", "dstest1_pgie_config.txt", NULL);

    /* we add a message handler */
    bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
    gst_object_unref (bus);

    /* Set up the pipeline */
    /* we add all elements into the pipeline */
#ifdef PLATFORM_TEGRA
    gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux,
      nvvidconv, nvosd, transform, sink, NULL);
#else
    gst_bin_add_many (GST_BIN (pipeline),
                      source, h264parser, decoder, streammux,
                      nvvidconv, nvosd, sink, NULL);
#endif

    GstPad *sinkpad, *srcpad;
    gchar pad_name_sink[16] = "sink_0";
    gchar pad_name_src[16] = "src";

    sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
    if (!sinkpad) {
        g_printerr ("Streammux request sink pad failed. Exiting.\n");
        return -1;
    }

    srcpad = gst_element_get_static_pad (decoder, pad_name_src);
    if (!srcpad) {
        g_printerr ("Decoder request src pad failed. Exiting.\n");
        return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
        g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
        return -1;
    }

    gst_object_unref (sinkpad);
    gst_object_unref (srcpad);

    /* we link the elements together */
    /* file-source -> h264-parser -> nvh264-decoder ->
     * nvinfer -> nvvidconv -> nvosd -> video-renderer */

    if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
        g_printerr ("Elements could not be linked: 1. Exiting.\n");
        return -1;
    }

#ifdef PLATFORM_TEGRA
    if (!gst_element_link_many (streammux,
      nvvidconv, nvosd,transform, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");

    return -1;
  }
#else
    if (!gst_element_link_many (streammux,
                                nvvidconv, nvosd, sink, NULL)) {
        g_printerr ("Elements could not be linked: 2. Exiting.\n");
        return -1;
    }
#endif

    /* Lets add probe to get informed of the meta data generated, we add probe to
     * the sink pad of the osd element, since by that time, the buffer would have
     * had got all the metadata. */
    osd_sink_pad = gst_element_get_static_pad (sink, "sink");
//    osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
    if (!osd_sink_pad)
        g_print ("Unable to get sink pad\n");
    else
        gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                           osd_sink_pad_buffer_probe, NULL, NULL);

    /* Set the pipeline to "playing" state */
    g_print ("Now playing: %s\n", argv[1]);
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait till pipeline encounters an error or EOS */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    /* Out of the main loop, clean up nicely */
    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);
    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);
    return 0;
}

