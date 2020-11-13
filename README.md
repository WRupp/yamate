# YaMate
### Yet-Another MATerial parameter Estimation toolbox
Provides helper functions for estimating material properties.

**yamate is currently in (very) early development.**

yamate uses scipy as an backend for optimization jobs. 

---
## Installation
clone this repository and install it by running

`pip install setup.py`

soon to be available in pypi.

---
## Examples
currently there are two examples to setup an mechanical uniaxial test on a _VariationalViscoHydrolysis_ material specimen.


---
### Compiling Fortran code
one may use [f2py](https://numpy.org/doc/stable/f2py/) to bind fortran and python code for better performance. The variational _VariationalViscoHydrolysis_ material  has some fortran routines in fortran as examples. f2py has some known limitations, such as no support for OOP in modern fortran. f2py works best in linux machines, but can be executed in windows as well if configured properly.

---
## Citing Yamate and Contributing


