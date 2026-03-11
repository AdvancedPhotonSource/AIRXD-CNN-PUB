from setuptools import setup, Extension, find_packages
import pybind11
import os

#Run this with: python setup_cpp_scripts.py build_ext --inplace

# Add OpenMP flags if available
extra_compile_args = ['-std=c++11', '-O3']  # Base optimization flags
extra_link_args = []

# Add OpenMP support (for parallel processing)
if os.name != 'nt':  # Unix-like systems
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
else:  # Windows
    extra_compile_args.append('/openmp')

# Define multiple extensions
ext_modules = [
    # Original normalization histogram extension
    Extension(
        'c_algorithms.normalization_histogram',  # Module name for import
        ['c_algorithms/normalization_histogram.cpp'],  # Source file
        include_dirs=[
            pybind11.get_include(),
            'c_algorithms'
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    
    # New logarithmic histogram extension
    Extension(
        'c_algorithms.logarithmic_histogram',  # Different module name
        ['c_algorithms/logarithmic_histogram.cpp'],  # New source file
        include_dirs=[
            pybind11.get_include(),
            'c_algorithms'
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        'c_algorithms.azimuthal_processing',  # Module name for import
        ['c_algorithms/azimuthal_processing.cpp'],  # Source file
        include_dirs=[
            pybind11.get_include(),
            'c_algorithms'
        ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='histogram_normalizer',
    version='0.2',
    description='Histogram normalization and logarithmic histogram tools',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    ext_modules=ext_modules,  # List of all extensions
    package_data={'': ['__init__.py']},
)