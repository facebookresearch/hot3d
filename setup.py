from setuptools import setup, find_packages

setup(
    name="hot3d",
    version="0.1.0",
    description="A dataset for 3D hand-object interaction built by Meta.",
    author="Preston Culbertson (of fork)",
    author_email="pculbertson@theaiinstitute.com",
    packages=find_packages(where="hot3d"),
    package_dir={"hot3d": "hot3d"},
    install_requires=[
        "projectaria_tools==1.5.2",
        "requests",
        "rerun-sdk",
        "vrs",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
