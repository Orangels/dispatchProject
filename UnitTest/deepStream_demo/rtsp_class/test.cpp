#include<cstdarg>
#include<iostream>
#include "dsHandler.h"
#include <thread>
#include <functional>

using namespace std;


//int pic_num = 0;
//queue<cv::Mat> imgQueue;
//mutex myMutex;
//condition_variable con_v_notification;
//
GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data,
        queue<cv::Mat> *imgQueue,mutex* myMutex,condition_variable* con_v_notification){

    GstBuffer *buf = (GstBuffer *) info->data;
    Mat frame = writeImage(buf);

    // 跳帧
//    int frame_number = pic_num;
//    pic_num++;
    std::unique_lock<std::mutex> guard(*myMutex);
//    if (pic_num % 2 == 0){
        cout << 1111 << endl;
        cout << imgQueue->size() << endl;
        imgQueue->push(frame);
        con_v_notification->notify_all();
        cout << 2222 << endl;
        guard.unlock();
//    }

    return GST_PAD_PROBE_OK;
}
//
//namespace {
//    std::function<GstPadProbeReturn(GstPad * , GstPadProbeInfo * , gpointer)> callback;
//    vector<std::function<GstPadProbeReturn(GstPad * , GstPadProbeInfo * , gpointer)>> callBackList;
////    extern "C"
//    GstPadProbeReturn wrapper (GstPad * pad, GstPadProbeInfo * info, gpointer u_data){
//        return callback(pad, info,  u_data);
//    }
//}

int main(int argc, char *argv[])
{
    int pic_num = 0;
    queue<cv::Mat> *imgQueue_0, *imgQueue_1;
    mutex *myMutex_0, *myMutex_1;
    condition_variable *con_v_notification_0, *con_v_notification_1;

//    callback = std::bind(&osd_sink_pad_buffer_probe, std::placeholders::_1,
//            std::placeholders::_2,std::placeholders::_3, &imgQueue, &myMutex, &con_v_notification);
//
    dsHandler dsHandler_ls_0("rtsp://172.16.104.175:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream",
                           1920,1080,4000000, 0, 1);

    dsHandler dsHandler_ls_1("rtsp://172.16.104.178:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream",
                           1920,1080,4000000, 1, 1);

    imgQueue_0 = &dsHandler_ls_0.imgQueue;
    myMutex_0 = &dsHandler_ls_0.myMutex;
    con_v_notification_0 = &dsHandler_ls_0.con_v_notification;

    imgQueue_1 = &dsHandler_ls_1.imgQueue;
    myMutex_1 = &dsHandler_ls_1.myMutex;
    con_v_notification_1 = &dsHandler_ls_1.con_v_notification;

    cout << &dsHandler_ls_0 << " " << &dsHandler_ls_1 << endl;
    cout << imgQueue_0 << " " << imgQueue_1 << endl;
    cout << myMutex_0 << " " << myMutex_1 << endl;
    cout << con_v_notification_0 << " " << con_v_notification_1 << endl;


    auto ProduceImage_0 = [&](){
//        dsHandler dsHandler_ls("rtsp://172.16.104.175:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream",
//                               1920,1080,4000000, wrapper, 1);
//        dsHandler dsHandler_ls("rtsp://172.16.104.175:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream",
//                               1920,1080,4000000, 1);
        dsHandler_ls_0.run();
    };

    auto ProduceImage_1 = [&](){
        dsHandler_ls_1.run();
    };

    auto ConsumeImage_0 = [&](){
        int num = 1000;
        while (true){

            std::unique_lock<std::mutex> guard(*myMutex_0);
            while(imgQueue_0->empty()) {
                std::cout << "Consumer is waiting for items...\n";
                con_v_notification_0->wait(guard);
            }
            cv::Mat img = imgQueue_0->front();
            imgQueue_0->pop();
            guard.unlock();

            string img_path;
            int frame_number = num;
            cout << " write img " << endl;
            img_path = "./imgs/" + to_string(frame_number) + ".jpg";
            cv::imwrite(img_path, img);
            num++;

        }
    };

    auto ConsumeImage_1 = [&](){
        int num = 1000;
        while (true){

            std::unique_lock<std::mutex> guard(*myMutex_1);
            while(imgQueue_1->empty()) {
                std::cout << "Consumer is waiting for items...\n";
                con_v_notification_1->wait(guard);
            }
            cv::Mat img = imgQueue_1->front();
            imgQueue_1->pop();
            guard.unlock();

            string img_path;
            int frame_number = num;
            cout << " write img " << endl;
            img_path = "./imgs_1/" + to_string(frame_number) + ".jpg";
            cv::imwrite(img_path, img);
            num++;

        }
    };

    thread thread_write_image_front_0(ProduceImage_0);
    thread thread_write_image_front_1(ProduceImage_1);
    thread thread_read_image_front_0(ConsumeImage_0);
    thread thread_read_image_front_1(ConsumeImage_1);

    thread_write_image_front_0.join();
    thread_read_image_front_0.join();
    thread_write_image_front_1.join();
    thread_read_image_front_1.join();


    return 0;
}