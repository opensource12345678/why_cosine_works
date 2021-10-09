# Eigencurve: Optimal Learning Rate Schedule for SGD on Quadratic Objectives with Skewed Hessian Spectrums

## Dependency

All code are implemented in Python. One can install the dependency as follows,

```sh
pip install -r requirements.txt
```

## Modified PyHessian

All other experiments use the generated eigenvalue distribution of Hessian,
which is already provided without the need to run PyHessian. This part of the
code is only for checking and reviewing, where we made some modifications
to make the commandline interface of PyHessian more flexible.

All codes are stored in branch "pyhessian", please checkout to that
branch first.

```sh
git checkout pyhessian
```

An alternative is to download the code of that branch to check the code.

```
https://github.com/opensource12345678/why_cosine_works/tree/pyhessian
```
