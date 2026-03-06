from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    import pybind11

    # Define the C++ extension
    ext_modules = [
        Pybind11Extension(
            "third_party.relnet.objective_functions.objective_functions_ext",
            ["third_party/relnet/objective_functions/objective_functions_ext.cpp"],
            libraries=["boost_system"],
            include_dirs=[
                "/usr/include/boost",
                pybind11.get_cmake_dir() + "/../../../include",
            ],
            cxx_std=11,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    # Fallback if pybind11 is not available
    ext_modules = []
    cmdclass = {}

setup(
    name="gnarl",
    version="0.1",
    packages=find_packages(include=["gnarl*", "third_party*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
)
