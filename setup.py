from setuptools import setup, find_packages

setup(
    name="mlci",
    version="0.1.0",
    description="Statistically rigorous ML model evaluation and benchmarking",
    author="Niccolò Forzano",
    author_email="nic.forz@gmail.com",
    url="https://github.com/nickforce989/mlci",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "scipy>=1.9",
        "scikit-learn>=1.1",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "pandas>=1.5",
        "joblib>=1.2",
        "tqdm>=4.64",
    ],
    extras_require={
        "bayesian": ["pymc>=5.0", "arviz>=0.15"],
        "torch": ["torch>=1.13"],
        "dev": ["pytest>=7.0", "pytest-cov", "black", "ruff"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
