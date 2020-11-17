import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="zkyhaxpy", # Replace with your own username
    version="0.2.7.9.1",
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
