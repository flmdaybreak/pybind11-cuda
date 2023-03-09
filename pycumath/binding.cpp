#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <Python.h>

#include <iomanip>
#include "./cpp/test.h"
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(_pycumath, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: _pycumath

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    pybind11::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const char *e) { //抓取const char* 类型异常，转换为python RuntimeError
            std::string errmsg("exception in c++: exception type: const char *, exception value: ");
            errmsg += e;
            PyErr_SetString(PyExc_RuntimeError, errmsg.c_str());
        } catch (const std::string &e) { //抓取const char* 类型异常，转换为python RuntimeError
            std::string errmsg("exception in c++: exception type: std::string, exception value: ");
            errmsg += e;
            PyErr_SetString(PyExc_RuntimeError, errmsg.c_str());
        } catch (const std::exception &e) { //抓取std::exception 类型异常，转换为python RuntimeError
            std::string errmsg("exception in c++: exception type: std::exception, exception value: ");
            errmsg += e.what();
            PyErr_SetString(PyExc_RuntimeError, errmsg.c_str());
        } catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "unknown exception in c++");
        }
    });
     m.def("multiply_with_scalar", 
	    map_array,py::call_guard<py::gil_scoped_release>());
    
    // m.def("multiply_with_scalar1", map_array1);
    // m.def("multiply_with_scalar", map_array);

    
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

