import setuptools
from pkg_resources import DistributionNotFound, get_distribution

INSTALL_DEPS = [
    "numpy>=1.17.3"
]


def get_dist(pkgname):
    try:
        return int(get_distribution(pkgname).version.split(".")[0])
    except DistributionNotFound:
        return None


def get_install_deps(deps=INSTALL_DEPS):
    tf_gpu_version = get_dist('tensorflow_gpu')
    if tf_gpu_version is None or tf_gpu_version < 2:
        deps.append('tensorflow>=2.0.0')
    return deps


with open("README.md", "r") as fh:
    long_description = "".join(fh.readlines()[1:])

setuptools.setup(
    name="seeker",
    version="1.0.0",
    author="Ayal B. Gussow, Noam Auslander",
    author_email="ayal.gussow@gmail.com, noamaus@gmail.com",
    description="Predict bacterial or phage sequence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://seeker.pythonanywhere.com",
    include_package_data=True,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=get_install_deps(),
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points = {
        'console_scripts': [
            'predict-metagenome=seeker.command_line:predict_metagenome',
        ],
    }
)
