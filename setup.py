import os
import sys
import platform
import subprocess
import re
import setuptools

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_version():
    version_file = read("pycumath/version.py")
    version_re = r"__version__ = \"(?P<version>.+)\""
    version = re.match(version_re, version_file).group("version")
    
    return version


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.1.0":
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        # todo remove later
        cfg = "Debug"
        build_args = ["--config", cfg]

        if platform.system() != "Windows":
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j4"]
        #else:
            #cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            #if sys.maxsize > 2 ** 32:
            #    cmake_args += ["-A", "x64"]
            #build_args += ["--", "/m"]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)


setuptools.setup(
    name="pycumath",
    version=find_version(),
    author="zichen",
    author_email="zichen@trustbe.net",
    description="A library for private set intersection on Python",
    license="Trustbe",
    keywords="psi ot hash",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    # url="https://github.com/OpenMined/TenSEAL",
    packages=setuptools.find_packages(include=["pycumath", "pycumath.*"]),
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Security :: Cryptography",
    ],
    ext_modules=[CMakeExtension("_pycuda_test")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
