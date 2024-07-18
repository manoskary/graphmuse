import setuptools
import os, sys
import numpy


dirname = os.path.dirname(__file__)
#
if os.name=='posix':
    print("Compiling for POSIX systems. . .")
    eca = [] # ["-std=c11"]
    eca.append("-DPOSIX")

    # from psutil import cpu_count
    #
    # thread_count = cpu_count(logical=False)
    # if thread_count > 1:
    #     eca.append(f"-DThread_Count_Arg={thread_count}")


elif sys.platform.startswith('win'):
    print("Compiling for Windows. . .")
    eca = [] # ["/std:c11"]
    eca.append("-DWindows")

else:
    eca = []
    # raise Exception("Unsupported OS, please use Linux or Windows.")

# add flag to turn off debug mode (increasing speed)
eca.append("-DGM_DEBUG_OFF")



ext_modules = [
    setuptools.Extension(
        name="graphmuse.samplers.csamplers", sources=[os.path.join("src", "gmsamplersmodule.c")], extra_compile_args = eca,
            extra_link_args = [], include_dirs=[os.path.join(numpy.get_include(), "numpy"), "include"])]

# os.environ["CC"] = "gcc"
# os.environ["CXX"] = "gcc"

long_description = open(os.path.join(os.path.dirname(__file__), 'README.md'), "r").read()

setuptools.setup(
    name='GraphMuse',
    version='0.0.1',
    description='GraphMuse is a Python Library for Graph Deep Learning on Symbolic Music.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    setup_requires=['numpy'],
    keywords=[
    "deep-learning",
    "symbolic-music",
    "pytorch",
    "geometric-deep-learning",
    "graph-neural-networks",
    "graph-convolutional-networks",
    ],
    classifiers=[
        "Development Status :: 0 - Gamma",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Graph Deep Learning",
    ],
    ext_modules=ext_modules,
    author='Emmanouil Karystinaios',
    maintainer='Emmanouil Karystinaios'
)