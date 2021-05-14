from setuptools import find_packages, setup

setup(
    name="ml_example",
    packages=find_packages(),
    version="0.1.0",
    description="HW1 for ml in prod",
    author="Pankratova Daria",
    install_requires=[
        "marshmallow==3.11.1",
        "marshmallow-dataclass==8.4.1",
        "PyYAML==5.3.1",
        "typing-extensions==3.7.4.3",
        "typing-inspect==0.6.0",
        "pandas==1.1.3",
        "scikit-learn==0.24.1",
        "numpy==1.19.2",
        "pickleshare==0.7.5",
        "json5==0.9.5",
        "jsonschema==3.2.0",
        "logger==1.4",
        "logging==0.4.9.6",
    ],
    license="MIT",
)
