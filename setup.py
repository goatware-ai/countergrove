from setuptools import setup, find_packages

setup(
    name="countergrove",
    version="0.0.2",
    url="https://github.com/goatware-ai/countergrove.git",
    author="Goatware AI",
    author_email="goatware.ai@gmail.com",
    description="Countergrove: causal learning from observational data with TensorFlow and Keras",
    packages=find_packages(),
    install_requires=[
        "pypress",
        "numpy >= 1.11.0",
        "tensorflow >= 2.11.0",
        "pandas >= 1.0.0",
        "tensorflow-addons>=0.15.0",
    ],
    dependency_links=["git+ssh://git@github.com/gmgeorg/pypress.git#egg=pypress-0.0.2"],
)
