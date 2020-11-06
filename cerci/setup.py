from distutils.core import setup, Extension

module = Extension("IncrementLibrary", sources=["test.cpp"])

setup(name="IncrementLibrary",
      version="1.0",
      description="An incrementation library for python.",
      ext_modules=[module])