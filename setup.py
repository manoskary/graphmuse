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
        name="graphmuse.samplers.csamplers", sources=[os.path.join("src", "gmsamplersmodule.c")], extra_compile_args = ["-fopenmp"],
            extra_link_args = ["-fopenmp"])]

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

from psutil import cpu_count

# this doesn't necessarily show the number of available cores for a process
# however since this is just setting a default value, the number of logical cores should be used
# for number of available cores, use len(psutil.Process().cpu_affinity())
os.environ["OMP_NUM_THREADS"] = str(cpu_count())


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