import numpy
from distutils.command.build_ext import build_ext
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from Cython.Compiler import Options


# Preserve `__doc__` on functions and classes
# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Options.docstrings = True

SLEEF_EXT = Extension(
    "thinc_sleef_ops.sleef",
    ["thinc_sleef_ops/sleef.pyx", "thinc_sleef_ops/array_scalar.cpp", "thinc_sleef_ops/array_sse.cpp", "thinc_sleef_ops/array_avx.cpp"],
    language="c++",
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-std=c++11"],
    extra_objects=["libsleef.a"]
)

COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
}
LINK_OPTIONS = {"msvc": ["-std=c++11"], "mingw32": ["-std=c++11"], "other": []}
COMPILER_DIRECTIVES = {
    "language_level": -3,
    "embedsignature": True,
    "annotation_typing": False,
}


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def setup_package():
    include_dirs = [
        numpy.get_include(),
    ]
    ext_modules = [SLEEF_EXT]
    print("Cythonizing sources")
    ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

    setup(
        name="thinc-sleef-ops",
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_subclass},
        package_data={"": ["*.pyx", "*.pxd", "*.pxi"]},
    )


if __name__ == "__main__":
    setup_package()
