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
        name="graphmuse.samplers.csamplers", sources=["src/gmsamplersmodule.c"],
            extra_link_args = [], include_dirs=[numpy.get_include(), "include"])]

# os.environ["CC"] = "gcc"
# os.environ["CXX"] = "gcc"

long_description = open(os.path.join(os.path.dirname(__file__), 'README.md'), "r").read()

setuptools.setup(
    name='graphmuse',
    version='0.0.1',
    description='GraphMuse is a Python Library for Graph Deep Learning on Symbolic Music.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    setup_requires=['numpy'],
    python_requires='>=3.10',
    install_requires=[
        "torch",
        "torch-geometric",
        "torch-sparse",
        "torch-scatter",
        "torch-cluster",
        "pyg-lib",
        "numpy>=1.21.0",
        "partitura>=1.5.0",
        "psutil==5.9.5",
    ],
    keywords=[
    "deep-learning",
    "symbolic-music",
    "pytorch",
    "geometric-deep-learning",
    "graph-neural-networks",
    "graph-convolutional-networks",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=ext_modules,
    author='Emmanouil Karystinaios',
    maintainer='Emmanouil Karystinaios'
)