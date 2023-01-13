# cython: language_level=3
import setuptools
import os
import numpy
from Cython.Build import cythonize

dirname = os.path.dirname(__file__)
graph = [os.path.join(dirname, "graphmuse", "utils", x) for x in ["cython_graph.pyx", "cython_utils.pyx", "cython_sampler.pyx"]]
# module = setuptools.Extension('graph', sources=[graph])
# module = cythonize(graph)
ext_modules = [
    setuptools.Extension(
        name="graphmuse.samplers.csamplers", sources=[os.path.join("src", "gmsamplersmodule.c")])]

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

print(numpy.get_include())

setuptools.setup(
    name='graphmuse',
    version='0.0.1',
    description='Graph Deep Learning for Music',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 0 - Gamma",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Graph Deep Learning",
    ],
    include_dirs=[os.path.join(numpy.get_include(),"numpy")],
    # ext_modules=[module],
    ext_modules= ext_modules,
    author='Emmanouil Karystinaios',
    maintainer='Emmanouil Karystinaios'
)