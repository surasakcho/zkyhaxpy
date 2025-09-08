import setuptools

PACKAGE_NAME = 'zkyhaxpy'
VERSION = '0.3.1.3.1'


with open("README.md", "r") as f:
    long_description = f.read()

with open(f"{PACKAGE_NAME}\__init__.py", "w") as f:
    f.write(f"__version__ = '{VERSION}'")

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="Surasak Choedpasuporn",
    author_email="surasak.cho@gmail.com",
    description="A swiss-knife Data Science package for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/surasakcho/zkyhaxpy",
    packages=setuptools.find_packages(),    
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

