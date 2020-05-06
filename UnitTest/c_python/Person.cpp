#include "Person.hpp"

Person::Person()
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

//        pFile = PyString_FromString("student");
        pFile = PyUnicode_FromString("student");
        pModule = PyImport_Import(pFile);
        if (!pModule)
        {
            printf("PyImport_Import student.py failed!\n");
            break;
        }

        m_pDict = PyModule_GetDict(pModule);
        if (!m_pDict)
        {
            printf("PyModule_GetDict student.py failed!\n");
            break;
        }

        pClass = PyDict_GetItemString(m_pDict, "Person");
        if (!pClass || !PyCallable_Check(pClass))
        {
            printf("PyDict_GetItemString Person failed!\n");
            break;
        }

//        m_pHandle = PyInstance_New(pClass, NULL, NULL);
        m_pHandle = PyObject_CallObject(pClass, nullptr);
        if (!m_pHandle)
        {
            printf("PyInstance_New Person failed!\n");
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

    printf("Person::Person() end!\n");
}

Person::~Person()
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
    printf("Person::~Person() end!\n");
}

void Person::Push(char *name, char *sex, int age)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
        PyObject_CallMethod(m_pHandle, "push", "ssi", name, sex, age);
    } while(0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}


void Person::Show()
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure(); //申请获取GIL
    Py_BEGIN_ALLOW_THREADS;
    Py_BLOCK_THREADS;

    do
    {
        PyObject_CallMethod(m_pHandle, "show", NULL, NULL);
    } while(0);

    Py_UNBLOCK_THREADS;
    Py_END_ALLOW_THREADS;
    PyGILState_Release(gstate);
}