"""Setuptools setup script."""
from setuptools import setup

setup(name='optical_lattice',
      version='1.0',
      description='Resolve atom occupations in an optical lattice',
      url='https://github.com/phys201/optical_lattice',
      author='Furkan Ozturk, Can Knaut, Erik Knall',
      author_email='erikknall@g.harvard.edu',
      license='GNU GPL',
      packages=['optical_lattice'],
      install_requires=[
            'matplotlib',
            'numpy',
            'Pillow==8.2.0',
            'pymc3==3.8',
            'scikit-image==0.16.2',
            'Theano==1.0.4',
            'scipy==1.4.1',
            'seaborn'

      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
