import setuptools

INSTALL_DEPS = [
    "numpy>=1.17.3",
    "tensorflow>=2.0.0"
]

with open("README.md", "r") as fh:
    long_description = "".join(fh.readlines()[1:])

setuptools.setup(
    name="seeker",
    version="1.0.3",
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
    install_requires=INSTALL_DEPS,
    test_suite='nose.collector',
    tests_require=['nose'],
    entry_points = {
        'console_scripts': [
            'predict-metagenome=seeker.command_line:predict_metagenome',
        ],
    }
)
