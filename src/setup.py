from setuptools import setup, find_packages

setup(
    name="prevelop",
    version="2.0.0",
    author="Kaspar Bunne",
    description="Python framework for clustering mixed-type manufacturing data",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "gower>=0.1",
        "hdbscan>=0.8",
        "matplotlib>=3.7",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scikit-learn-extra>=0.3",
        "scipy>=1.11",
        "seaborn>=0.12",
        "openpyxl>=3.1",
    ],
    extras_require={
        "dashboard": [
            "streamlit>=1.30",
            "plotly>=5.18",
        ],
    },
    include_package_data=True,
)
