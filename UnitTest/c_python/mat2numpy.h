#include <Python.h>
#include <numpy/arrayobject.h>
#include <opencv2/imgproc/imgproc.hpp>


void mat2np(cv::Mat img, PyObject* ArgList);
void list2vector(PyObject* pyResult,std::vector<int> &vret);