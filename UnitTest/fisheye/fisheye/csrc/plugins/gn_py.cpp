//
// Created by 李大冲 on 2019-11-01.
//


#include "GN.h"
#include "NvInfer.h"
//#include "NvCaffeParser.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(gnplugin, m)
{
namespace py = pybind11;

// This allows us to use the bindings exposed by the tensorrt module.
py::module::import("tensorrt");

// Note that we only need to bind the constructors manually. Since all other methods override IPlugin functionality, they will be automatically available in the python bindings.
// The `std::unique_ptr<GNPlugin, py::nodelete>` specifies that Python is not responsible for destroying the object. This is required because the destructor is private.
py::class_<GNPlugin, nvinfer1::IPluginExt, std::unique_ptr<GNPlugin, py::nodelete>>(m, "GNPlugin")
// Bind the normal constructor as well as the one which deserializes the plugin
.def(py::init<const nvinfer1::Weights*, int>())
.def(py::init<const void*, size_t>())
;

// Our custom plugin factory derives from both nvcaffeparser1::IPluginFactoryExt and nvinfer1::IPluginFactory
py::class_<GNPluginFactory,   nvinfer1::IPluginFactory>(m, "GNPluginFactory")
// Bind the default constructor.
.def(py::init<>())
// The destroy_plugin function does not override either of the base classes, so we must bind it explicitly.
.def("destroy_plugin", &GNPluginFactory::destroyPlugin)
;
}
