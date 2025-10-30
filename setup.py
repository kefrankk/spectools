from setuptools import setup, find_packages

setup(
    name="spectools",  # nome do pacote
    version="0.1.0",   # versão inicial
    author="Kelly Heckler",
    author_email="kelly.heckler@acad.ufsm.br",
    description="Spectroscopic data analysis tools",
    url="https://github.com/kefrankk/spectools",  # link do repositório
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # código dentro da pasta src/
    python_requires=">=3.12",
)
