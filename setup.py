import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GridOverview", # Replace with your own username
    version="0.0.1",
    author="Frank Backs",
    author_email="backsfpa@gmail.com",
    description="4D griddata plotting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FPABacks/OverviewPlot/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)

