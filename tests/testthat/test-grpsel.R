test_that('sample sizes must be equal', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(9)
  expect_error(grpsel(x, y))
})

test_that('group must have ncol(x) elements', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, group = 1:11))
})

test_that('group (as a list) must have ncol(x) elements', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, group = list(1:5, 6:11)))
})

test_that('y must be in {0,1} with logistic loss', {
  set.seed(123)
  y <- rbinom(10, 1, 0.5)
  y[y == 0] <- - 1
  x <- rnorm(10)
  expect_error(grpsel(x, y, loss = 'logistic'))
})

test_that('y must not contain NAs', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  y[1] <- NA
  expect_error(grpsel(x, y))
})

test_that('x must not contain NAs', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  x[1] <- NA
  expect_error(grpsel(x, y))
})

test_that('lambda must contain a vector for every unique gamma', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, penalty = 'grSubset+grLasso', gamma = 1:5, lambda = 0))
})

test_that('lambda must contain ngamma vectors', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = 0))
})

test_that('nlambda must be greater than zero', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, nlambda = 0))
})

test_that('ngamma must be greater than zero', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, ngamma = 0))
})

test_that('alpha must be in (0,1)', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, alpha = 1))
})

test_that('when max.cd.iter is exceeded a warning is provided', {
  set.seed(123)
  y <- rnorm(100)
  x <- matrix(rnorm(100 * 10), 100, 10)
  expect_warning(grpsel(x, y, max.cd.iter = 5))
})

test_that('when max.ls.iter is exceeded a warning is provided', {
  set.seed(123)
  y <- rnorm(100)
  x <- matrix(rnorm(100 * 10), 100, 10)
  expect_warning(grpsel(x, y, max.ls.iter = 0))
})

test_that('gamma.min must be positive', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, gamma.min = 0))
})

test_that('gamma.max must be positive', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, gamma.max = 0))
})

test_that('subset.factor must have same length as group', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, subset.factor = rep(1, 11)))
})

test_that('lasso.factor must have same length as group', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, lasso.factor = rep(1, 11)))
})

test_that('ridge.factor must have same length as group', {
  set.seed(123)
  y <- rnorm(10)
  x <- rnorm(10)
  expect_error(grpsel(x, y, ridge.factor = rep(1, 11)))
})

test_that('regression with square loss works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with logistic loss works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y, group, loss = 'logistic', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'binomial')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with different sized groups works', {
  set.seed(123)
  group <- c(1, 1, 1, 1, 2, 2, 2, 3, 3, 4)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with overlapping groups works', {
  set.seed(123)
  group <- list(1:5, 5:10)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression without orthogonalisation works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, orthogonalise = F)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression without sorting works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, sort = F)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with screening works', {
  set.seed(123)
  group <- rep(1:15, each = 2)
  x <- matrix(rnorm(100 * 30), 100, 30)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, screen = 5)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with screening and violations works', {
  set.seed(123)
  group <- rep(1:15, each = 2)
  x <- matrix(rnorm(100 * 30), 100, 30)
  y <- rowSums(x) + rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, screen = 1)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with a constant response works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rep(1, 100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, family = 'gaussian')$coef)
  beta.target[2] <- 0
  expect_equal(beta, beta.target)
})

test_that('coefficients are extracted correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x, family = 'gaussian')
  beta <- coef(fit, lambda = 0)
  beta.target <- as.matrix(as.numeric(coef(fit.target)))
  expect_equal(beta, beta.target)
})

test_that('predictions are computed correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x, family = 'gaussian')
  yhat <- predict(fit, x, lambda = 0)
  yhat.target <- as.matrix(as.numeric(predict(fit.target, as.data.frame(x))))
  expect_equal(yhat, yhat.target)
})

test_that('predictions are computed correctly when x is a data frame', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x, family = 'gaussian')
  yhat <- predict(fit, as.data.frame(x), lambda = 0)
  yhat.target <- as.matrix(as.numeric(predict(fit.target, as.data.frame(x))))
  expect_equal(yhat, yhat.target)
})

test_that('plot function returns a plot', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  p <- plot(fit)
  expect_s3_class(p, 'ggplot')
})

test_that('number of predictors does not exceed pmax', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, pmax = 5)
  beta <- coef(fit)[, ncol(coef(fit))]
  sparsity <- sum(beta[- 1] != 0)
  expect_lte(sparsity, 5)
})

test_that('number of groups does not exceed gmax', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, loss = 'square', eps = 1e-15, gmax = 2)
  beta <- coef(fit)[, ncol(coef(fit))]
  group.sparsity <- sum(vapply(unique(group), \(k) norm(beta[- 1][which(group == k)], '2'),
                               numeric(1)) != 0)
  expect_lte(group.sparsity, 2)
})

test_that('number of ridge solutions is ngamma', {
  set.seed(123)
  n <- 100
  p <- 10
  gamma <- 0.1
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  fit <- grpsel(x, y, penalty = 'grSubset+Ridge', ngamma = 5, lambda = rep(list(0), 5), eps = 1e-15)
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma', {
  set.seed(123)
  n <- 100
  p <- 10
  gamma <- 0.1
  x <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = rep(list(0), 5),
                eps = 1e-15)
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma for logistic loss', {
  set.seed(123)
  n <- 100
  p <- 10
  gamma <- 0.1
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', loss = 'logistic', ngamma = 5,
                lambda = rep(list(0), 5), eps = 1e-15)
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma for logistic loss', {
  set.seed(123)
  n <- 100
  p <- 10
  gamma <- 0.1
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', loss = 'logistic', ngamma = 5,
                lambda = rep(list(0), 5), lasso.factor = c(0, rep(1, 9)), eps = 1e-15)
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma', {
  set.seed(123)
  n <- 100
  p <- 10
  gamma <- 0.1
  x <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = rep(list(0), 5),
                lasso.factor = c(0, rep(1, 9)), eps = 1e-15)
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('local search improves on coordinate descent for logistic loss', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + 2 * matrix(rnorm(n), n, p)
  y <- rbinom(n, 1, 1 / (1 + exp(- rowSums(x[, 1:10]))))
  fit.ls <- grpsel(x, y, group, loss = 'logistic', ls = T)
  fit.cd <- grpsel(x, y, group, loss = 'logistic', ls = F)
  loss.ls <- fit.ls$loss[[1]][which(fit.ls$np[[1]] %in% fit.cd$np[[1]])]
  loss.cd <- fit.cd$loss[[1]][which(fit.cd$np[[1]] %in% fit.ls$np[[1]])]
  expect_true(sum(loss.ls) < sum(loss.cd))
})

test_that('local search improves on coordinate descent for square loss', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + 2 * matrix(rnorm(n), n, p)
  y <- rnorm(n, rowSums(x[, 1:10]))
  fit.ls <- grpsel(x, y, group, loss = 'square', ls = T)
  fit.cd <- grpsel(x, y, group, loss = 'square', ls = F)
  loss.ls <- fit.ls$loss[[1]][which(fit.ls$np[[1]] %in% fit.cd$np[[1]])]
  loss.cd <- fit.cd$loss[[1]][which(fit.cd$np[[1]] %in% fit.ls$np[[1]])]
  expect_true(sum(loss.ls) < sum(loss.cd))
})

test_that('local search improves on coordinate descent for square loss without orthogonalisation', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + 2 * matrix(rnorm(n), n, p)
  y <- rnorm(n, rowSums(x[, 1:10]))
  fit.ls <- grpsel(x, y, group, loss = 'square', ls = T, orthogonalise = F)
  fit.cd <- grpsel(x, y, group, loss = 'square', ls = F, orthogonalise = F)
  loss.ls <- fit.ls$loss[[1]][which(fit.ls$np[[1]] %in% fit.cd$np[[1]])]
  loss.cd <- fit.cd$loss[[1]][which(fit.cd$np[[1]] %in% fit.ls$np[[1]])]
  expect_true(sum(loss.ls) < sum(loss.cd))
})
