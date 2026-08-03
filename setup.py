from setuptools import setup, find_packages

setup(
    name="protint",
    version="0.1.0",
    description="ProtInt: A deep learning model for integrating proteomics data from cell lines and tumors",
    author="Cong Quan Ta, Ursula Klingmüller, Andreas Raue",
    author_email="cong.ta@uni-a.de",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "anndata",
        "scanpy",
        "mlflow"
    ],
    keywords="protint",
    entry_points={
        'console_scripts': ['protint = protint.main:main']
    }
)