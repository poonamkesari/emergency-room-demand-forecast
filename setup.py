from setuptools import find_packages, setup

setup(
    name="er_demand_forecasting",
    version="0.1.0",
    description="Short-term emergency department demand forecasting pipeline.",
    author="Data Science Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas==3.0.2",
        "numpy==2.4.4",
        "matplotlib==3.10.9",
        "seaborn==0.13.2",
        "scikit-learn==1.8.0",
        "statsmodels==0.14.6",
        "lightgbm==4.6.0",
        "kaggle==1.7.4.5",
        "PyYAML",
    ],
    python_requires=">=3.11",
)
