import contextlib
import platform
from pathlib import Path
import sysconfig
import shutil
from setuptools import setup
from setuptools.command import build_ext

# bazel build extensions taken from https://github.com/google/benchmark/blob/main/setup.py

PYTHON_INCLUDE_PATH_PLACEHOLDER = "<PYTHON_INCLUDE_PATH>"

@contextlib.contextmanager
def temp_fill_include_path(fp: str):
    """Temporarily set the Python include path in a file."""
    with open(fp, "r+") as f:
        try:
            content = f.read()
            replaced = content.replace(
                PYTHON_INCLUDE_PATH_PLACEHOLDER,
                Path(sysconfig.get_paths()['include']).as_posix(),
            )
            f.seek(0)
            f.write(replaced)
            f.truncate()
            yield
        finally:
            # revert to the original content after exit
            f.seek(0)
            f.write(content)
            f.truncate()


class BazelExtension(setuptools.Extension):
    """A C/C++ extension that is defined as a Bazel BUILD target."""

    def __init__(self, name: str, bazel_target: str):
        super().__init__(name=name, sources=[])

        self.bazel_target = bazel_target
        stripped_target = bazel_target.split("//")[-1]
        self.relpath, self.target_name = stripped_target.split(":")

class BuildBazelExtension(build_ext.build_ext):
    """A command that runs Bazel to build a C/C++ extension."""

    def run(self):
        for ext in self.extensions:
            self.bazel_build(ext)
        build_ext.build_ext.run(self)

    def bazel_build(self, ext: BazelExtension):
        """Runs the bazel build to create the package."""
        CWD=os.getcwd()
        os.chdir('gbbs')
        with temp_fill_include_path("WORKSPACE"):
            #temp_path = Path(self.build_temp)

            bazel_argv = [
                "bazel",
                "build",
                ext.bazel_target,
                #f"--symlink_prefix={temp_path / 'bazel-'}",
                f"--compilation_mode={'dbg' if self.debug else 'opt'}",
            ]

            self.spawn(bazel_argv)

            # explicitly call `bazel shutdown` for graceful exit
            self.spawn(["bazel", "shutdown"])

        os.chdir(CWD)

def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "sparsecluster",
    "version": "0.0.1",
    "description": "sparse hiearchical agglomerative clustering with pynndescent and gbbs SeqHAC",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: C",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    "keywords": "HAC, nearest neighbor, knn, ANN",
    "url": "http://github.com/bobermayer/sparsecluster",
    "author": "Benedikt Obermayer",
    "author_email": "benedikt.obermayer@bih-charite.de",
    "maintainer": "Benedikt Obermayer",
    "maintainer_email": "benedikt.obermayer@bih-charite.de",
    "license": "BSD",
    "packages": ["sparsecluster"],
    "install_requires": [
        "scikit-learn >= 0.18",
        "scipy >= 1.0",
        "numba >= 0.51.2",
        "llvmlite >= 0.30",
        "joblib >= 0.11",
        'importlib-metadata >= 4.8.1; python_version < "3.8"',
    ],
    "cmdclass" : {"build_ext": BuildBazelExtension},
    "ext_modules": [BazelExtension("gbbs_lib", "//pybindings:gbbs_lib")],
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
