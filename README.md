# why_cosine_works

## Dependency

All code are implemented in Python. One can install the dependency as follows,

```sh
pip install -r requirements.txt
```

## Ridge Regression in a4a

All codes are stored in branch "ridge-regression", please checkout to that
branch first.

```sh
git checkout ridge-regression
```

An alternative is to download the code of that branch to run this part of the
experiments.

```
https://github.com/opensource12345678/why_cosine_works/tree/ridge_regression
```

Two scripts are used for generating the ridge regression results.
`ridge_regression.sh` provides the grid search process. After it finished, run
`collect_ridge_regression_stat.sh` to collect info from log files.

```sh
./ridge_regression.sh
./collect_ridge_regression_stat.sh 1
./collect_ridge_regression_stat.sh 5
./collect_ridge_regression_stat.sh 25
./collect_ridge_regression_stat.sh 250
```
