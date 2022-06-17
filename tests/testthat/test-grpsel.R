test_that('sample sizes must be equal', {
  set.seed(123)
  x <- rnorm(9)
  y <- rnorm(10)
  expect_error(grpsel(x, y))
})

test_that('group must have ncol(x) elements', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, group = 1:11))
})

test_that('group (as a list) must have ncol(x) elements', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, group = list(1:5, 6:11)))
})

test_that('y must be in {0,1} with logistic loss', {
  set.seed(123)
  x <- rnorm(10)
  y <- rbinom(10, 1, 0.5)
  y[y == 0] <- - 1
  expect_error(grpsel(x, y, loss = 'logistic'))
})

test_that('y must not contain NAs', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  y[1] <- NA
  expect_error(grpsel(x, y))
})

test_that('x must not contain NAs', {
  set.seed(123)
  x <- rnorm(10)
  x[1] <- NA
  y <- rnorm(10)
  expect_error(grpsel(x, y))
})

test_that('lambda must contain a vector for every unique gamma', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, penalty = 'grSubset+grLasso', gamma = 1:5, lambda = 0))
})

test_that('lambda must contain ngamma vectors', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = 0))
})

test_that('nlambda must be greater than zero', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, nlambda = 0))
})

test_that('ngamma must be greater than zero', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, ngamma = 0))
})

test_that('lambda.step must be in (0,1)', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, lambda.step = 1))
})

test_that('when max.cd.iter is exceeded a warning is provided', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  expect_warning(grpsel(x, y, max.cd.iter = 0))
})

test_that('when max.ls.iter is exceeded a warning is provided', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  expect_warning(grpsel(x, y, max.ls.iter = 0))
})

test_that('gamma.min must be positive', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, gamma.min = 0))
})

test_that('gamma.max must be positive', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, gamma.max = 0))
})

test_that('lambda.factor must have same length as group', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, lambda.factor = rep(1, 11)))
})

test_that('gamma.factor must have same length as group', {
  set.seed(123)
  x <- rnorm(10)
  y <- rnorm(10)
  expect_error(grpsel(x, y, gamma.factor = rep(1, 11)))
})

test_that('maximum group size cannot exceed sample size when orthogonalising', {
  set.seed(123)
  group <- c(rep(1, 6), rep(2, 4))
  x <- matrix(rnorm(5 * 10), 5, 10)
  y <- rnorm(5)
  expect_error(grpsel(x, y, group, orthogonalise = T))
})

test_that('square loss regression works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('logistic loss regression works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y, group, loss = 'logistic', eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x, 'binomial')$coef)
  expect_equal(beta, beta.target)
})

test_that('different sized groups work', {
  set.seed(123)
  group <- c(1, 1, 1, 1, 2, 2, 2, 3, 3, 4)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('overlapping groups work', {
  set.seed(123)
  group <- list(1:5, 5:10)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('orthogonalisation works', {
  set.seed(123)
  group <- c(1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15, orthogonalise = T)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with overlapping groups and orthogonalisation works', {
  set.seed(123)
  group <- list(1:5, 5:10)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, orthogonalise = T, eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('regression without sorting works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15, sort = F)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with screening works', {
  set.seed(123)
  group <- rep(1:15, each = 2)
  x <- matrix(rnorm(100 * 30), 100, 30)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15, screen = 5)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with screening and violations works', {
  set.seed(123)
  group <- rep(1:15, each = 2)
  x <- matrix(rnorm(100 * 30), 100, 30)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, eps = 1e-15, screen = 1)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('regression with a constant response works', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rep(1, 100)
  fit <- grpsel(x, y, group, eps = 1e-15)
  beta <- coef(fit)[, ncol(coef(fit))]
  beta.target <- as.numeric(glm(y ~ x)$coef)
  expect_equal(beta, beta.target)
})

test_that('coefficients are extracted correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x)
  beta <- coef(fit, lambda = 0)
  beta.target <- as.matrix(as.numeric(coef(fit.target)))
  expect_equal(beta, beta.target)
})

test_that('predictions are computed correctly', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x)
  yhat <- predict(fit, x, lambda = 0)
  yhat.target <- as.matrix(as.numeric(predict(fit.target, as.data.frame(x))))
  expect_equal(yhat, yhat.target)
})

test_that('predictions are computed correctly when x is a data frame', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, eps = 1e-15)
  fit.target <- glm(y ~ x)
  yhat <- predict(fit, as.data.frame(x), lambda = 0)
  yhat.target <- as.matrix(as.numeric(predict(fit.target, as.data.frame(x))))
  expect_equal(yhat, yhat.target)
})

test_that('plot function returns a plot', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y)
  p <- plot(fit)
  expect_s3_class(p, 'ggplot')
})

test_that('number of predictors does not exceed pmax', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, pmax = 5)
  beta <- coef(fit)[, ncol(coef(fit))]
  sparsity <- sum(beta[- 1] != 0)
  expect_lte(sparsity, 5)
})

test_that('number of groups does not exceed gmax', {
  set.seed(123)
  group <- rep(1:5, each = 2)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, group, gmax = 2)
  beta <- coef(fit)[, ncol(coef(fit))]
  group.sparsity <- sum(vapply(unique(group), \(k) norm(beta[- 1][which(group == k)], '2'),
                               numeric(1)) != 0)
  expect_lte(group.sparsity, 2)
})

test_that('number of ridge solutions is ngamma with square loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, penalty = 'grSubset+Ridge', ngamma = 5, lambda = rep(list(0), 5))
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of ridge solutions is ngamma with logistic loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+Ridge', loss = 'logistic', ngamma = 5,
                lambda = rep(list(0), 5))
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma with square loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = rep(list(0), 5))
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('number of group lasso solutions is ngamma with logistic loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', loss = 'logistic', ngamma = 5,
                lambda = rep(list(0), 5))
  beta <- coef(fit)
  expect_equal(ncol(beta), 5)
})

test_that('unpenalised group subset coefficients are always nonzero with logistic loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y,  loss = 'logistic', nlambda = 5, lambda.factor = c(0, rep(1, 9)))
  beta <- coef(fit)
  expect_equal(sum(beta[2, ] != 0), 5)
})

test_that('unpenalised group lasso coefficients are always nonzero with square loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rnorm(100)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', ngamma = 5, lambda = rep(list(0), 5),
                gamma.factor = c(0, rep(1, 9)))
  beta <- coef(fit)
  expect_equal(sum(beta[2, ] != 0), 5)
})

test_that('unpenalised group lasso coefficients are always with logistic loss', {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), 100, 10)
  y <- rbinom(100, 1, 0.5)
  fit <- grpsel(x, y, penalty = 'grSubset+grLasso', loss = 'logistic', ngamma = 5,
                lambda = rep(list(0), 5), gamma.factor = c(0, rep(1, 9)))
  beta <- coef(fit)
  expect_equal(sum(beta[2, ] != 0), 5)
})

test_that('local search improves on coordinate descent for square loss', {
  set.seed(123)
  group <- rep(1:10, each = 2)
  x <- matrix(rnorm(100 * 20), 100, 20) + matrix(rnorm(100), 100, 20)
  y <- rnorm(100, rowSums(x[, 1:10]))
  fit.ls <- grpsel(x, y, group, local.search = T)
  fit.cd <- grpsel(x, y, group)
  loss.ls <- fit.ls$loss[[1]][fit.ls$np[[1]] %in% fit.cd$np[[1]]]
  loss.cd <- fit.cd$loss[[1]][fit.cd$np[[1]] %in% fit.ls$np[[1]]]
  expect_true(sum(loss.ls) < sum(loss.cd))
})

test_that('local search improves on coordinate descent for square loss with orthogonalisation', {
  set.seed(123)
  group <- rep(1:10, each = 2)
  x <- matrix(rnorm(100 * 20), 100, 20) + matrix(rnorm(100), 100, 20)
  y <- rnorm(100, rowSums(x[, 1:10]))
  fit.ls <- grpsel(x, y, group, local.search = T, orthogonalise = T)
  fit.cd <- grpsel(x, y, group, orthogonalise = T)
  loss.ls <- fit.ls$loss[[1]][fit.ls$np[[1]] %in% fit.cd$np[[1]]]
  loss.cd <- fit.cd$loss[[1]][fit.cd$np[[1]] %in% fit.ls$np[[1]]]
  expect_true(sum(loss.ls) < sum(loss.cd))
})

test_that('local search improves on coordinate descent for logistic loss', {
  set.seed(123)
  group <- rep(1:10, each = 2)
  x <- matrix(rnorm(100 * 20), 100, 20) + matrix(rnorm(100), 100, 20)
  y <- rbinom(100, 1, 1 / (1 + exp(- rowSums(x[, 1:10]))))
  fit.ls <- grpsel(x, y, group, loss = 'logistic', local.search = T)
  fit.cd <- grpsel(x, y, group, loss = 'logistic')
  loss.ls <- fit.ls$loss[[1]][fit.ls$np[[1]] %in% fit.cd$np[[1]]]
  loss.cd <- fit.cd$loss[[1]][fit.cd$np[[1]] %in% fit.ls$np[[1]]]
  expect_true(sum(loss.ls) < sum(loss.cd))
})
