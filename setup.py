from setuptools import setup

setup(
    name='PairwiseDistances',
    version='1.0',
    author='Oliver K. Ernst',
    packages=['pairwiseDistances'],
    install_requires=[
          'numpy'
      ],
    license='GNU General Public License v3.0',
    description='Calculate pairwise distances between particles',
    long_description=open('README.md').read(),
    url="https://github.com/smrfeld/PairwiseDistances",
     classifiers=[
         "Development Status :: 5 - Production/Stable",
         "Intended Audience :: Developers",
         "Topic :: Scientific/Engineering :: Mathematics",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     package_data = {
        'examples': ['*.md']
    }
)
