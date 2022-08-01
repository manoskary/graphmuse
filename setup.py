# cython: language_level=3
import setuptools
import os
import numpy
from Cython.Build import cythonize

dirname = os.path.dirname(__file__)
graph = os.path.join(dirname, "graphmuse", "utils", "create_graph.pyx")
# module = setuptools.Extension('graph', sources=[graph])
module = cythonize(graph)

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

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
    include_dirs=[numpy.get_include()],
    # ext_modules=[module],
    ext_modules= module,
    author='Emmanouil Karystinaios',
    maintainer='Emmanouil Karystinaios'
)