import os
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

if sys.platform == "win32":
    flags = ["/O2", "/std:c++17"]
else:
    flags = ["-O3", "-std=c++17", "-DNDEBUG"]
    if os.environ.get("DAGGER_NATIVE") == "1":
        flags.append("-march=native")

extensions = [
    Extension("dagger." + name, ["src/dagger/" + name + ".pyx"],
              language="c++", extra_compile_args=flags)
    for name in ("_core", "_training")
]

setup(ext_modules=cythonize(extensions, compiler_directives={
    "language_level": 3, "boundscheck": False, "wraparound": False,
    "initializedcheck": False, "cdivision": True,
}))

