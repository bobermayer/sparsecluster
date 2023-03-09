from setuptools import setup
from setuptools.command import build_ext


PYTHON_INCLUDE_PATH_PLACEHOLDER = "<PYTHON_INCLUDE_PATH>"

IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

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
        with temp_fill_include_path("WORKSPACE"):
            temp_path = Path(self.build_temp)

            bazel_argv = [
                "bazel",
                "build",
                ext.bazel_target,
                f"--symlink_prefix={temp_path / 'bazel-'}",
                f"--compilation_mode={'dbg' if self.debug else 'opt'}",
            ]

            if IS_WINDOWS:
                # Link with python*.lib.
                for library_dir in self.library_dirs:
                    bazel_argv.append("--linkopt=/LIBPATH:" + library_dir)
            elif IS_MAC:
                if platform.machine() == "x86_64":
                    # C++17 needs macOS 10.14 at minimum
                    bazel_argv.append("--macos_minimum_os=10.14")

                    # cross-compilation for Mac ARM64 on GitHub Mac x86 runners.
                    # ARCHFLAGS is set by cibuildwheel before macOS wheel builds.
                    archflags = os.getenv("ARCHFLAGS", "")
                    if "arm64" in archflags:
                        bazel_argv.append("--cpu=darwin_arm64")
                        bazel_argv.append("--macos_cpus=arm64")

                elif platform.machine() == "arm64":
                    bazel_argv.append("--macos_minimum_os=11.0")

            self.spawn(bazel_argv)

            shared_lib_suffix = '.dll' if IS_WINDOWS else '.so'
            ext_name = ext.target_name + shared_lib_suffix
            ext_bazel_bin_path = temp_path / 'bazel-bin' / ext.relpath / ext_name

            ext_dest_path = Path(self.get_ext_fullpath(ext.name))
            shutil.copyfile(ext_bazel_bin_path, ext_dest_path)

            # explicitly call `bazel shutdown` for graceful exit
            self.spawn(["bazel", "shutdown"])


def readme():
    with open("README.rst") as readme_file:
        return readme_file.read()


configuration = {
    "name": "sparsecluster",
    "version": "0.0.1dev",
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
    "ext_modules": [BazelBuildExtension("gbbs/gbbs_lib", "//gbbs:pybindings:gbbs_lib.so")],
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": False,
}

setup(**configuration)
