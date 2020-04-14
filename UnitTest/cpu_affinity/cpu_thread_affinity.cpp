//
// Created by Orangels on 2020-04-14.
//
#include <unistd.h>
#ifndef __USE_GNU
#define __USE_GNU   // 为了使用SET_SET()等宏定义，但unistd.h里面好像已经定义了
#endif
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <vector>

unsigned int systemCPUNum()
{
    // _SC_NPROCESSORS_CONF的值为CPU个数，基于0开始编号
    return sysconf(_SC_NPROCESSORS_CONF);
}

bool currentProcessAffinity(std::vector<unsigned int>& runningCPUVector)
{
    cpu_set_t cpuSet;

    // 清空一个CPU集合
    CPU_ZERO(&cpuSet);

    // 得到指定进程ID绑定到哪个CPU
    int ret = sched_getaffinity(0, // 0代表当前进程
                                sizeof(cpuSet),
                                &cpuSet);
    if (ret < 0)
    {
        return false;
    }

    unsigned int cpuNum = systemCPUNum();
    runningCPUVector.clear();
    for (unsigned int i = 0; i < cpuNum; ++i)
    {
        // 检查一个CPU号是否在一个集合中
        if (CPU_ISSET(i, &cpuSet))
        {
            runningCPUVector.push_back(i);
        }
    }

    return true;
}

bool setCurrentProcessAffinity(const std::vector<unsigned int>& needBindCPUVector)
{
    cpu_set_t cpuSet;

    // 清空一个CPU集合
    CPU_ZERO(&cpuSet);

    for (auto& iter : needBindCPUVector)
    {
        CPU_SET(iter, &cpuSet);
    }

    // 将指定进程ID绑定到CPU
    int ret = sched_setaffinity(0, // 0代表当前进程
                                sizeof(cpuSet),
                                &cpuSet);
    if (ret < 0)
    {
        return false;
    }

    return true;
}

bool currentThreadAffinity(std::vector<unsigned int>& runningCPUVector)
{
    cpu_set_t cpuSet;

    // 清空一个CPU集合
    CPU_ZERO(&cpuSet);

    // 得到指定线程ID绑定到哪个CPU
    int ret = pthread_getaffinity_np(pthread_self(),
                                     sizeof(cpuSet),
                                     &cpuSet);
    if (ret < 0)
    {
        return false;
    }

    unsigned int cpuNum = systemCPUNum();
    runningCPUVector.clear();
    for (unsigned int i = 0; i < cpuNum; ++i)
    {
        // 检查一个CPU号是否在一个集合中
        if (CPU_ISSET(i, &cpuSet))
        {
            runningCPUVector.push_back(i);
        }
    }

    return true;
}

bool setCurrentThreadAffinity(const std::vector<unsigned int>& needBindCPUVector)
{
    cpu_set_t cpuSet;

    // 清空一个CPU集合
    CPU_ZERO(&cpuSet);

    for (auto& iter : needBindCPUVector)
    {
        CPU_SET(iter, &cpuSet);
    }

    // 将指定线程ID绑定到CPU
    int ret = pthread_setaffinity_np(pthread_self(),
                                     sizeof(cpuSet),
                                     &cpuSet);
    if (ret < 0)
    {
        return false;
    }

    return true;
}

int main()
{
    printf("*****Process bind CPU sample*****\n");
    unsigned int cpuNum = systemCPUNum();
    printf("Current system has %u CPU(s)\n", cpuNum);

    std::vector<unsigned int> runningCPUVector;
    if (!currentProcessAffinity(runningCPUVector))
    {
        printf("Get current process was bound witch CPU failed\n");
        return 1;
    }

    for (auto& iter : runningCPUVector)
    {
        printf("Current process is running at %u CPU\n", iter);
    }

    std::vector<unsigned int> needBindCPUVector {0, 2};
    if (!setCurrentProcessAffinity(needBindCPUVector))
    {
        printf("Current process bind CPU failed\n");
        return 1;
    }

    printf("Current process bind CPU success\n");

    runningCPUVector.clear();
    if (!currentProcessAffinity(runningCPUVector))
    {
        printf("Get current process was bound witch CPU failed\n");
        return 1;
    }

    for (auto& iter : runningCPUVector)
    {
        printf("Current process is running at %u CPU\n", iter);
    }

    printf("\n*****Thread bind CPU sample*****\n");
    runningCPUVector.clear();
    if (!currentThreadAffinity(runningCPUVector))
    {
        printf("Get current thread was bound witch CPU failed\n");
        return 1;
    }

    for (auto& iter : runningCPUVector)
    {
        printf("Thread %lu is running at %u CPU\n", pthread_self(), iter);
    }

    needBindCPUVector.clear();
    needBindCPUVector.push_back(1);
    if (!setCurrentThreadAffinity(needBindCPUVector))
    {
        printf("Current thread bind CPU failed\n");
        return 1;
    }

    printf("Thread %lu bind CPU success\n", pthread_self());

    runningCPUVector.clear();
    if (!currentThreadAffinity(runningCPUVector))
    {
        printf("Get current thread was bound witch CPU failed\n");
        return 1;
    }

    for (auto& iter : runningCPUVector)
    {
        printf("Thread %lu is running at %u CPU\n", pthread_self(), iter);
    }

    return 0;
}