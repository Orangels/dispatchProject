#include "EnginePy.hpp"
#include "mat2numpy.h"
#include <iostream>

Engine_api::Engine_api()
{
    PyObject* pFile = NULL;
    PyObject* pModule = NULL;
    PyObject* pClass = NULL;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
#if 0
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            printf("Py_Initialize error!\n");
            break;
        }
#endif

        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('../')");

        pFile = PyUnicode_FromString("engine_api");
        pModule = PyImport_Import(pFile);
        if (!pModule)
        {
            printf("PyImport_Import Engine_api.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict)
        {
            printf("PyModule_GetDict Engine_api.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "ObjectApi");
        if (!pClass || !PyCallable_Check(pClass))
        {
            printf("PyDict_GetItemString Engine_api failed!\n");
            break;
        }

        m_pHandle = PyObject_CallObject(pClass, nullptr);
        if (!m_pHandle)
        {
            printf("PyInstance_New ObjectApi failed!\n");
            break;
        }
    } while (0);

    if (pClass)
        Py_DECREF(pClass);
    //if (m_pDict)
    //       Py_DECREF(m_pDict);
    if (pModule)
        Py_DECREF(pModule);
    if (pFile)
        Py_DECREF(pFile);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

    printf("Engine_api::Engine_api() end!\n");
}

Engine_api::~Engine_api()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    if (m_pHandle)
        Py_DECREF(m_pHandle);
    if (m_pDict)
        Py_DECREF(m_pDict);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);

#if 0
    Py_Finalize();
#endif
    printf("EnginePy::~EnginePy() end!\n");
}

void Engine_api::get_result(Mat frame)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
        PyObject *ArgList1 = PyTuple_New(1);
        mat2np(frame, ArgList1);
        std::cout << "py infer start" << std::endl;
        PyObject *pyResult1 = PyObject_CallMethod(m_pHandle,"get_result","O",ArgList1);
        std::cout << "py infer end" << std::endl;
    } while(0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}

void Engine_api::test(){
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
        PyObject_CallMethod(m_pHandle, "test", NULL, NULL);
    } while(0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}