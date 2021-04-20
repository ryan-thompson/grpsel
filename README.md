
# grpsel

## Overview

An R package for sparse regression modelling with grouped predictors
(including overlapping groups). `grpsel` makes use of the group subset
selection penalty, which usually leads to excellent selection and
prediction. Optionally, the group subset penalty can be combined with a
group lasso or ridge penalty for added shrinkage. Linear and logistic
regression are currently supported.

## Installation

To install `grpsel` from GitHub, run the following code:

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

# Fit the group subset regularisation path
fit <- grpsel(x, y, group)
coef(fit, lambda = 0.05)[, ]
```

    ##  [1] 0.1363218 1.0738569 0.9734314 0.8432187 1.1940502 0.0000000 0.0000000
    ##  [8] 0.0000000 0.0000000 0.0000000 0.0000000

``` r
# Cross-validate the group subset regularisation path
fit <- cv.grpsel(x, y, group)
coef(fit)[, ]
```

    ##  [1] 0.1363218 1.0738569 0.9734314 0.8432187 1.1940502 0.0000000 0.0000000
    ##  [8] 0.0000000 0.0000000 0.0000000 0.0000000

Check out the package vignette for more help getting started.
