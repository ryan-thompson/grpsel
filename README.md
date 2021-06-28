
# grpsel

[![R-CMD-check](https://github.com/ryan-thompson/grpsel/workflows/R-CMD-check/badge.svg)](https://github.com/ryan-thompson/grpsel/actions)
[![codecov](https://codecov.io/gh/ryan-thompson/grpsel/branch/master/graph/badge.svg)](https://github.com/ryan-thompson/grpsel/actions)

## Overview

An R package for sparse regression modelling with grouped predictors
(including overlapping groups). `grpsel` uses the group subset selection
penalty, which usually leads to excellent selection and prediction.
Optionally, the group subset penalty can be combined with a group lasso
or ridge penalty for added shrinkage. Linear and logistic regression are
currently supported. See [this paper](https://arxiv.org/abs/2105.12081)
for more information.

## Installation

To install the latest stable version from CRAN, run the following code:

``` r
install.packages('grpsel')
```

To install the latest development version from GitHub, run the following
code:

``` r
devtools::install_github('ryan-thompson/grpsel')
```

## Usage

The `grpsel()` function fits a group subset regression model for a
sequence of tuning parameters. The `cv.grpsel()` function provides a
convenient way to automatically cross-validate these parameters.

``` r
library(grpsel)

# Generate some grouped data
set.seed(123)
n <- 100 # Number of observations
p <- 10 # Number of predictors
g <- 5 # Number of groups
group <- rep(1:g, each = p / g) # Group structure
beta <- numeric(p)
beta[which(group %in% 1:2)] <- 1 # First two groups are nonzero
x <- matrix(rnorm(n * p), n, p)
y <- x %*% beta + rnorm(n)

# Fit the group subset selection regularisation path
fit <- grpsel(x, y, group)
coef(fit, lambda = 0.05)
```

    ##            [,1]
    ##  [1,] 0.1363218
    ##  [2,] 1.0738569
    ##  [3,] 0.9734314
    ##  [4,] 0.8432187
    ##  [5,] 1.1940502
    ##  [6,] 0.0000000
    ##  [7,] 0.0000000
    ##  [8,] 0.0000000
    ##  [9,] 0.0000000
    ## [10,] 0.0000000
    ## [11,] 0.0000000

``` r
# Cross-validate the group subset selection regularisation path
fit <- cv.grpsel(x, y, group)
coef(fit)
```

    ##            [,1]
    ##  [1,] 0.1363218
    ##  [2,] 1.0738569
    ##  [3,] 0.9734314
    ##  [4,] 0.8432187
    ##  [5,] 1.1940502
    ##  [6,] 0.0000000
    ##  [7,] 0.0000000
    ##  [8,] 0.0000000
    ##  [9,] 0.0000000
    ## [10,] 0.0000000
    ## [11,] 0.0000000

## Documentation

See the package
[vignette](https://CRAN.R-project.org/package=grpsel/vignettes/vignette.html)
or [reference
manual](https://CRAN.R-project.org/package=grpsel/grpsel.pdf).
