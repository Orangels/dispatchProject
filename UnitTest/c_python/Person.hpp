#ifndef __PERSON_HPP__
#define __PERSON_HPP__

#include <Python.h>
#include <stdio.h>
#include <time.h>

#define LOG_DEBUG(msg, ...) printf("[%s][%s][%s][%d]: " msg, __TIME__, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)





class Person
{
public:
    PyObject* m_pDict = NULL;
    PyObject* m_pHandle = NULL;
public:
    Person();
    ~Person();
    void Push(char *name, char *sex, int age);
    void Show();

};

#endif