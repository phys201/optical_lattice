"""Setuptools setup script."""
from setuptools import setup

setup(name='optical_lattice',
      version='0.1',
      description='Resolve atom occupations in an optical lattice',
      url='https://github.com/phys201/optical_lattice',
      author='Furkan Ozturk, Can Knaut, Erik Knall',
      author_email='erikknall@g.harvard.edu',
      license='GNU GPL',
      packages=['optical_lattice'],
      install_requires=[
            'matplotlib',
            'numpy',
            'Pillow',
            'pymc3',
            'scikit-image'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
