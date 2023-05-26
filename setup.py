import io
import os
from typing import List
from setuptools import find_packages, setup

def read(*paths, **kwargs) -> str:
    content = ''
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get('encoding', 'utf8')
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path) -> List[str]:
    return [
        line.strip()
        for line in read(path).split('\n')
        if not line.startswith(('"', '#', '-', 'git+'))
    ]


setup(
    name='occts',
    version=read('occts', 'VERSION'),
    description='One-Class classification algorithms for time series',
    url='https://github.com/jose-gilberto/icdm-23-occts',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    author='Gilberto Barbosa',
    packages=find_packages(exclude=['tests', '.github']),
    install_requires=read_requirements('requirements.txt'),
    entry_points={
        'console_scripts': ['occts = occts.__main__:main']
    },
    extras_require={'test': read_requirements('requirements-test.txt')}
)

