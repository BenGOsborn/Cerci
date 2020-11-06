#include <Python.h>

int CUDAincrement(int n) {
    return n+1;
};

static PyObject* increment(PyObject* self, PyObject* args) {
    int n;

    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;

    return Py_BuildValue("i", CUDAincrement(n));
};

static PyObject* version(PyObject* self) {
    return Py_BuildValue("s", "Version 1.0");
};

static PyMethodDef myMethods[] = {
    {"increment", increment, METH_VARARGS, "Returns the incremented value of which it is given."},
    {"version", (PyCFunction)version, METH_NOARGS, "Returns the version of the library."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef IncrementLibrary = {
    PyModuleDef_HEAD_INIT,
    "IncrementLibrary",
    "Incrementation module.",
    -1,
    myMethods
};

PyMODINIT_FUNC PyInit_IncrementLibrary(void) {
    return PyModule_Create(&IncrementLibrary);
};