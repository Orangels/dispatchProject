#include<cstdarg>
#include<iostream>
#include "dsHandler.h"
#include <thread>

using namespace std;

int add(int pre,...){
    va_list arg_ptr;

    int sum=0;
    int nArgValue;

    sum+=pre;

    va_start(arg_ptr,pre);
    do
    {
        nArgValue=va_arg(arg_ptr,int);
        sum+=nArgValue;

    }while(nArgValue!=0);   //自定义结束条件是输入参数为0

    va_end(arg_ptr);

    return sum;
}

int dsHandler::pic_num = 0;
queue<cv::Mat> dsHandler::imgQueue;
mutex dsHandler::myMutex;
condition_variable dsHandler::con_v_notification;

int main(int argc, char *argv[])
{
//    cout<<add(1,2,3,0)<<endl;  //必须以0结尾，因为参数列表结束的判断条件是读到0停止

    auto ProduceImage = [&](){
        dsHandler dsHandler_ls("rtsp://admin:sx123456@192.168.88.38:554/h264/ch2/sub/av_stream",
                               1920,1080,4000000);
        dsHandler_ls.run();
    };

    auto ConsumeImage = [&](){
        while (true){

            std::unique_lock<std::mutex> guard(dsHandler::myMutex);
            while(dsHandler::imgQueue.empty()) {
                std::cout << "Consumer is waiting for items...\n";
                dsHandler::con_v_notification.wait(guard);
            }
            cv::Mat img = dsHandler::imgQueue.front();
            dsHandler::imgQueue.pop();
            guard.unlock();

            string img_path;
            int frame_number = dsHandler::pic_num;
            img_path = "./imgs/" + to_string(frame_number) + ".jpg";
            cv::imwrite(img_path, img);

        }
    };

    thread thread_write_image_front(ProduceImage);
    thread thread_read_image_front(ConsumeImage);

    thread_write_image_front.join();
    thread_read_image_front.join();


    return 0;
}