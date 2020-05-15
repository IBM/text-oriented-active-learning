from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='toal',
    version='0.0.42',
    description='text oriented active learning',
    long_description=readme,
    author='Pierpaolo Tommasi, Charles Jochim',
    author_email='ptommasi@ie.ibm.com, ',
    url='https://github.com/IBM/text-oriented-active-learning',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
