# cython: language_level=3
import setuptools
import os, sys
import numpy


dirname = os.path.dirname(__file__)

if os.name=='posix':
    eca = ["-std=c11"]
    eca.append("-DPOSIX")

    #add flag to turn off debug mode (increasing speed)
    #eca.append("-DGM_DEBUG_OFF")

    from psutil import cpu_count
    thread_count = cpu_count(logical=False)

    if thread_count>1:
        eca.append(f"-DThread_Count_Arg={thread_count}")
elif sys.platform.startswith('win'):
    eca = ["-DWindows"]

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
    include_dirs=[os.path.join(numpy.get_include(), "numpy"), "include", "../miniconda3/include/libxml2/libxml"],
    # ext_modules=[module],
    ext_modules= ext_modules,
    author='Emmanouil Karystinaios, Nimrod Varga',
    maintainer='Emmanouil Karystinaios, Nimrod Varga'
)