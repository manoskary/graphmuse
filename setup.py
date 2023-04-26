thread_count =  1
# cython: language_level=3
import setuptools
import os
import numpy
from Cython.Build import cythonize
from psutil import cpu_count

dirname = os.path.dirname(__file__)
graph = [os.path.join(dirname, "graphmuse", "utils", x) for x in ["cython_graph.pyx", "cython_utils.pyx", "cython_sampler.pyx"]]
# module = setuptools.Extension('graph', sources=[graph])
# module = cythonize(graph)

eca = ["-std=c11"]

if os.name=='posix':
    eca.append("-DPOSIX")

if thread_count>1:
    eca.append(f"-DThread_Count_Arg={thread_count}")

ext_modules = [
    setuptools.Extension(
        name="graphmuse.samplers.csamplers", sources=[os.path.join("src", "gmsamplersmodule.c")], extra_compile_args = eca,
            extra_link_args = [])]

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"






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
    include_dirs=[os.path.join(numpy.get_include(),"numpy"), "include"],
    # ext_modules=[module],
    ext_modules= ext_modules,
    author='Emmanouil Karystinaios',
    maintainer='Emmanouil Karystinaios'
)