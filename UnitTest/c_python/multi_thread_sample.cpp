/***************************************
  g++ main.cpp -o main -I/usr/include/python2.7/ -lpython2.7 -lpthread
 ****************************************/

#include "Person.hpp"
#include <time.h>
#include <pthread.h>

#define WAIT_COUNT 10
#define MAX_SIZE 20
Person *pAppBuf[MAX_SIZE] = {NULL};
pthread_t ThreadBuf[MAX_SIZE] = {0};
int nCurSize = 0;

pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;


void *thread_func(void *args)
{
    Person *pApp = (Person *)args;

    int count = WAIT_COUNT;
    while(count > 0)
    {
        count--;
        //pthread_mutex_lock(&mut);
        pApp->Push("jack", "man", count);
        //pthread_mutex_unlock(&mut);
        sleep(1);
    }

    pthread_exit(NULL);
}


void multi_thread_create(int n, Person *pPer)
{
    int ret = 0;
    if (n > MAX_SIZE)
    {
        return;
    }

    for (int i = 0; i < n; i++)
    {
        pthread_t threadid;
        ret = pthread_create(&threadid, NULL, thread_func, pPer);
        if (ret != 0)
        {
            LOG_DEBUG("pthread_create failed!\n");
            break;
        }
        ThreadBuf[i] = threadid;
        nCurSize++;
    }

    LOG_DEBUG("thread_create end\n");
}


void multi_thread_destory()
{
    LOG_DEBUG("nCurSize = %d\n", nCurSize);
    for (int i = 0; i < nCurSize; ++i)
    {
        LOG_DEBUG("pthread_join %d thread\n", i);
        pthread_t threadid = ThreadBuf[i];
        pthread_join(threadid, NULL);
    }
}


int main()
{
    int ret = 0;
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        LOG_DEBUG("Py_Initialize error, return\n");
        return -1;
    }

    PyEval_InitThreads();
    int nInit = PyEval_ThreadsInitialized();
    if (nInit)
    {
        LOG_DEBUG("PyEval_SaveThread\n");
        PyEval_SaveThread();
    }

    Person *pPer = new Person();

    multi_thread_create(5, pPer);

    int count = WAIT_COUNT;
    while (count > 0)
    {
        count--;
        //pthread_mutex_lock(&mut);
        pPer->Show();
        printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
        //pthread_mutex_unlock(&mut);
        sleep(1);
    }

    multi_thread_destory();

    delete pPer;

    PyGILState_STATE gstate = PyGILState_Ensure();
    Py_Finalize();
    LOG_DEBUG("main end\n");
    return 0;
}