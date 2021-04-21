set.seed(123)

test_that('sample sizes must be equal', {
  y <- rnorm(10)
  x <- rnorm(9)
  expect_error(grpsel(x, y))
})

test_that('unpenalized regression with square loss works', {
  x <- matrix(rnorm(20 * 5), 20, 5)
  y <- rnorm(20)
  beta <- grpsel(x, y, loss = 'square', lambda = 0, eps = 1e-15)$beta[[1]]
  beta.target <- matrix(glm(y ~ x, family = 'gaussian')$coef, 6, 1)
  expect_equal(beta, beta.target)
})

test_that('unpenalized regression with logistic loss works', {
  x <- matrix(rnorm(20 * 5), 20, 5)
  y <- rbinom(20, 1, 0.5)
  beta <- grpsel(x, y, loss = 'logistic', lambda = 0, eps = 1e-15)$beta[[1]]
  beta.target <- matrix(glm(y ~ x, family = 'binomial')$coef, 6, 1)
  expect_equal(beta, beta.target)
})

