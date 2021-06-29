test_that('nfold must be valid', {
  set.seed(123)
  y <- rnorm(100)
  x <- rnorm(100)
  expect_error(cv.grpsel(x, y, nfold = 1))
  expect_error(cv.grpsel(x, y, nfold = 101))
})

test_that('length of folds must match sample size', {
  set.seed(123)
  y <- rnorm(100)
  x <- rnorm(100)
  folds <- sample(2, 101, T)
  expect_error(cv.grpsel(x, y, folds = folds))
})

test_that('cross-validation leads to the correct subset under square loss', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x[, which(group %in% 1)]) + rnorm(100)
  fit <- cv.grpsel(x, y, group, loss = 'square', eps = 1e-15)
  beta <- as.numeric(coef(fit))
  beta.target <- rep(0, 11)
  beta.target[c(1, which(group %in% 1) + 1)] <- as.numeric(glm(y ~ x[, which(group %in% 1)],
                                                               family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('cross-validation leads to the correct subset under logistic loss', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 1 / (1 + exp(- rowSums(x[, which(group %in% 1)]))))
  fit <- cv.grpsel(x, y, group, loss = 'logistic', eps = 1e-15)
  beta <- as.numeric(coef(fit))
  beta.target <- rep(0, 11)
  beta.target[c(1, which(group %in% 1) + 1)] <- as.numeric(glm(y ~ x[, which(group %in% 1)],
                                                               family = 'binomial')$coef)
  expect_equal(beta, beta.target)
})

test_that('cross-validation works when folds are manually supplied', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x[, which(group %in% 1)]) + rnorm(100)
  folds <- sample(2, 100, T)
  fit <- cv.grpsel(x, y, group, loss = 'square', eps = 1e-15, folds = folds)
  beta <- as.numeric(coef(fit))
  beta.target <- rep(0, 11)
  beta.target[c(1, which(group %in% 1) + 1)] <- as.numeric(glm(y ~ x[, which(group %in% 1)],
                                                               family = 'gaussian')$coef)
  expect_equal(beta, beta.target)
})

test_that('coefficients are extracted correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- cv.grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x, family = 'gaussian')
  beta <- coef(fit)
  beta.target <- as.matrix(as.numeric(coef(fit.target)))
  expect_equal(beta, beta.target)
})

test_that('predictions are computed correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- cv.grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x, family = 'gaussian')
  yhat <- predict(fit, x)
  yhat.target <- as.matrix(as.numeric(predict(fit.target, as.data.frame(x))))
  expect_equal(yhat, yhat.target)
})

test_that('plot function returns a plot', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  fit <- cv.grpsel(x, y, eps = 1e-15)
  p <- plot(fit)
  expect_s3_class(p, 'ggplot')
})

test_that('local search improves on coordinate descent for logistic loss', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + matrix(rnorm(n), n, p)
  y <- rbinom(n, 1, 1 / (1 + exp(- rowSums(x[, 1:4]))))
  fit <- cv.grpsel(x, y, group, loss = 'logistic', ls = T)
  beta <- coef(fit)
  expect_true(identical(which(beta[- 1] != 0), 1:4))
})

test_that('local search improves on coordinate descent for square loss', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + matrix(rnorm(n), n, p)
  y <- rnorm(n, rowSums(x[, 1:4]))
  fit <- cv.grpsel(x, y, group, loss = 'square', ls = T)
  beta <- coef(fit)
  expect_true(identical(which(beta[- 1] != 0), 1:4))
})

test_that('local search improves on coordinate descent for square loss with orthogonalisation', {
  set.seed(123)
  n <- 100
  p <- 50
  group <- rep(1:25, each = 2)
  x <- matrix(rnorm(n * p), n, p) + matrix(rnorm(n), n, p)
  y <- rnorm(n, rowSums(x[, 1:4]))
  fit <- cv.grpsel(x, y, group, loss = 'square', ls = T, orthogonalise = F)
  beta <- coef(fit)
  expect_true(identical(which(beta[- 1] != 0), 1:4))
})

test_that('sequential and parallel cross-validation produce same output', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rowSums(x) + rnorm(100)
  folds <- rep(1:10, each = 10)
  fit.seq <- cv.grpsel(x, y, eps = 1e-15, folds = folds)
  cl <- parallel::makeCluster(2)
  fit.par <- cv.grpsel(x, y, eps = 1e-15, folds = folds, cluster = cl)
  parallel::stopCluster(cl)
  expect_equal(fit.seq, fit.par)
})
