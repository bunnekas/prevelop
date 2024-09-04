from setuptools import setup, find_packages

setup(
    name="PrEvelOp",
    version="1.0.4",
    author="Kaspar Bunne",
    author_email="kaspar.bunne@fir.rwth-aachen.de",
    description="PrEvelOp - Production Development Optimization",
    url="https://innolab-git.fir.de/innolab-management/im/prevelop",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "pandas>=2.2.2",
        "scikit_learn>=1.3.2",
        "seaborn>=0.13.2",
        "matplotlib>=3.8.2",
        "numpy>=1.26.4",
        "gower>=0.1.2",
        "scipy>=1.14.0",
        "yellowbrick>=1.5",
        "numpy>=2.0.1",
        "openpyxl>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "cool-script=my_cool_project.cli:main",
        ],
    },
    include_package_data=True,
)
