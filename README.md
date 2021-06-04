# why_cosine_works

## Dependency

All code are implemented in Python. One can install the dependency as follows,

```sh
pip install -r requirements.txt
```

## CIFAR-10 with 10 Epochs

All codes are stored in branch "cifar10-10-epochs", please checkout to that
branch first.

```sh
git checkout cifar10-10-epochs
```

An alternative is to download the code of that branch to run this part of the
experiments.

```
https://github.com/opensource12345678/why_cosine_works/tree/cifar10-10-epochs
```

### Grid Search Based on Validation Accuracy

This part of the code requires the support of GPU.

```sh
# Validation scripts that performs grid search
./cifar10_resnet18_val.sh
./cifar10_googlenet_val.sh
./cifar10_vgg16_val.sh

# Collect results with best validation accuracy
./collect_val_results.sh
```

### Test on Searched Hyperparameters

`cifar10_resnet18_test.sh`/`cifar10_googlenet_test.sh`/`cifar10_vgg16_test.sh`
have almost same structures as the validation scripts. Modify the corresponding
hyperparameters in the script and run,

```
# Test scripts that test the model on searched hyperparameters
./cifar10_resnet18_test.sh
./cifar10_googlenet_test.sh
./cifar10_vgg16_test.sh

# Collect test results
./collect_test_results.sh
```

Please contact the author if there are any questions.


## CIFAR-10 with 100 Epochs

All codes are stored in branch "cifar10-100-epochs", please checkout to that
branch first.

```sh
git checkout cifar10-100-epochs
```

An alternative is to download the code of that branch to run this part of the
experiments.

```
https://github.com/opensource12345678/why_cosine_works/tree/cifar10-100-epochs
```

Other parts are exactly the same as "CIFAR-10 with 10 Epochs".


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

## Modified PyHessian

All aboved experiments use the generated eigenvalue distribution of Hessian,
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
