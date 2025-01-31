from setuptools import setup, find_packages

setup(
    name="PrEvelOp",
    version="1.0.5",
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
        "gower==0.1.2",
        "hdbscan==0.8.40",
        "matplotlib==3.9.3",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit_learn==1.5.2",
        "scikit_learn_extra==0.3.0",
        "scipy==1.14.1",
        "seaborn==0.13.2",
        "setuptools==68.1.2",
        "yellowbrick==1.5",
        "openpyxl==3.1.5",
    ],
    entry_points={
        "console_scripts": [
            "cool-script=my_cool_project.cli:main",
        ],
    },
    include_package_data=True,
)
