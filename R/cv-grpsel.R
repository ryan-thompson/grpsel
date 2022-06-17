#' @title Cross-validated group subset selection
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Fits the regularisation surface for a regression model with a group subset selection
#' penalty and then cross-validates this surface.
#'
#' @param x a predictor matrix
#' @param y a response vector
#' @param group a vector of length \code{ncol(x)} with the jth element identifying the group that
#' the jth predictor belongs to; alternatively, a list of vectors with the kth vector identifying
#' the predictors that belong to the kth group (useful for overlapping groups)
#' @param penalty the type of penalty to apply; one of 'grSubset', 'grSubset+grLasso', or
#' 'grSubset+Ridge'
#' @param loss the type of loss function to use; 'square' for linear regression or 'logistic' for
#' logistic regression
#' @param lambda an optional list of decreasing sequences of group subset selection parameters; the
#' list should contain a vector for each value of \code{gamma}
#' @param gamma an optional decreasing sequence of group lasso or ridge parameters
#' @param nfold the number of cross-validation folds
#' @param folds an optional vector of length \code{nrow(x)} with the ith entry identifying the fold
#' that the ith observation belongs to
#' @param cv.loss an optional cross-validation loss-function to use; should accept a vector of
#' predicted values and a vector of actual values
#' @param cluster an optional cluster for running cross-validation in parallel; must be set up using
#' \code{parallel::makeCluster}; each fold is evaluated on a different node of the cluster
#' @param interpolate a logical indicating whether to interpolate the \code{lambda} sequence for
#' the cross-validation fits; see details below
#' @param ... any other arguments for \code{grpsel()}
#'
#' @details When \code{loss='logistic'} stratified cross-validation is used to balance
#' the folds. When fitting to the cross-validation folds, \code{interpolate=TRUE} cross-validates
#' the midpoints between consecutive \code{lambda} values rather than the original \code{lambda}
#' sequence. This new sequence retains the same set of solutions on the full data, but often leads
#' to superior cross-validation performance.
#'
#' @return An object of class \code{cv.grpsel}; a list with the following components:
#' \item{cv.mean}{a list of vectors containing cross-validation means per value of \code{lambda};
#' an individual vector in the list for each value of \code{gamma}}
#' \item{cd.sd}{a list of vectors containing cross-validation standard errors per value of
#' \code{lambda}; an individual vector in the list for each value of \code{gamma}}
#' \item{lambda}{a list of vectors containing the values of \code{lambda} used in the fit; an
#' individual vector in the list for each value of \code{gamma}}
#' \item{gamma}{a vector containing the values of \code{gamma} used in the fit}
#' \item{lambda.min}{the value of \code{lambda} minimising \code{cv.mean}}
#' \item{gamma.min}{the value of \code{gamma} minimising \code{cv.mean}}
#' \item{fit}{the fit from running \code{grpsel()} on the full data}
#'
#' @example R/examples/example-cv-grpsel.R
#'
#' @export

cv.grpsel <- \(x, y, group = seq_len(ncol(x)),
               penalty = c('grSubset', 'grSubset+grLasso', 'grSubset+Ridge'),
               loss = c('square', 'logistic'), lambda = NULL, gamma = NULL, nfold = 10,
               folds = NULL, cv.loss = NULL, cluster = NULL, interpolate = TRUE, ...) {

  penalty <- match.arg(penalty)
  loss <- match.arg(loss)

  # Check data is valid
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.matrix(y)) y <- as.matrix(y)

  # Check arguments are valid
  if (nfold < 2 | nfold > nrow(x)) {
    stop('nfolds must be at least 2 and at most the number of rows in x')
  }
  if (!is.null(folds) & length(folds) != nrow(x)) {
    stop('length of folds must equal number of rows in x')
  }

  # Set up for cross-validation
  lambda.compute <- is.null(lambda)
  fit <- grpsel(x, y, group, penalty, loss, lambda = lambda, gamma = gamma, ...)
  lambda <- fit$lambda
  gamma <- fit$gamma
  nlambda <- vapply(lambda, length, integer(1))
  ngamma <- length(fit$gamma)
  if (is.null(folds)) {
    if (loss == 'square') {
      folds <- sample(rep_len(1:nfold, nrow(x)))
    } else if (loss == 'logistic') {
      folds <- integer(nrow(x))
      folds[y == 0] <- sample(rep_len(1:nfold, sum(y == 0)))
      folds[y == 1] <- sample(rep_len(1:nfold, sum(y == 1)))
    }
  } else {
    nfold <- length(unique(folds))
  }

  # Save cross-validation loss functions
  if (is.null(cv.loss)) {
    if (loss == 'square') {
      cv.loss <- \(xb, y) 0.5 * mean((y - xb) ^ 2)
    } else if (loss == 'logistic') {
      cv.loss <- \(xb, y) {
        pi <- pmax(1e-5, pmin(1 - 1e-5, 1 / (1 + exp(- xb))))
        - mean(y * log(pi) + (1 - y) * log(1 - pi))
      }
    }
  }

  # If lambda was computed, use midpoints between consecutive lambdas in cross-validation
  lambda.cv <- lambda
  if (interpolate & lambda.compute) {
    for (i in 1:ngamma) {
      if (length(lambda[[i]]) > 2) {
        lambda.cv[[i]][2:(nlambda[i] - 1)] <-
          lambda[[i]][- c(1, nlambda[i])] + diff(lambda[[i]][- 1]) / 2
      }
    }
  }

  # Loop over folds
  cvf <- \(fold) {
    fold.ind <- which(folds == fold)
    x.train <- x[- fold.ind, , drop = FALSE]
    x.valid <- x[fold.ind, , drop = FALSE]
    y.train <- y[- fold.ind, , drop = FALSE]
    y.valid <- y[fold.ind, , drop = FALSE]
    fit.fold <- grpsel(x.train, y.train, group, penalty, loss, lambda = lambda.cv, gamma = gamma,
                       ...)
    cv <- list()
    for (i in 1:ngamma) {
      cv[[i]] <- apply(predict(fit.fold, x.valid, gamma = gamma[i]), 2, cv.loss, y.valid)
    }
    cv
  }
  if (is.null(cluster)) {
    cv <- lapply(1:nfold, cvf)
  } else {
    parallel::clusterCall(cluster, \() library(grpsel))
    cv <- parallel::clusterApply(cluster, 1:nfold, cvf)
  }
  cv <- lapply(1:ngamma, \(i) t(simplify2array(lapply(cv, `[[`, i))))

  # Compose cross-validation results
  cv.mean <- lapply(cv, colMeans)
  cv.sd <- lapply(cv, \(x) apply(x, 2, stats::sd) / sqrt(nfold))
  gamma.min.ind <- which.min(vapply(cv.mean, min, numeric(1)))
  gamma.min <- gamma[gamma.min.ind]
  lambda.min.ind <- which.min(cv.mean[[gamma.min.ind]])
  lambda.min <- lambda[[gamma.min.ind]][lambda.min.ind]

  # Return result
  result <- list(cv.mean = cv.mean, cv.sd = cv.sd, lambda = lambda, gamma = gamma,
                 lambda.min = lambda.min, gamma.min = gamma.min, fit = fit)
  class(result) <- 'cv.grpsel'
  return(result)

}

#==================================================================================================#
# Coefficient function
#==================================================================================================#

#' @title Coefficient function for cv.grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Extracts coefficients for specified values of the tuning parameters.
#'
#' @param object an object of class \code{cv.grpsel}
#' @param lambda the value of \code{lambda} indexing the desired fit
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A matrix of coefficients.
#'
#' @method coef cv.grpsel
#'
#' @export
#'
#' @importFrom stats "coef"

coef.cv.grpsel <- \(object, lambda = 'lambda.min', gamma = 'gamma.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  if (!is.null(gamma)) if (gamma == 'gamma.min') gamma <- object$gamma.min
  coef.grpsel(object$fit, lambda = lambda, gamma = gamma, ...)

}

#==================================================================================================#
# Predict function
#==================================================================================================#

#' @title Predict function for cv.grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Generate predictions for new data using specified values of the tuning parameters.
#'
#' @param object an object of class \code{cv.grpsel}
#' @param x.new a matrix of new values for the predictors
#' @param lambda the value of \code{lambda} indexing the desired fit
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A matrix of predictions.
#'
#' @method predict cv.grpsel
#'
#' @export
#'
#' @importFrom stats "predict"

predict.cv.grpsel <- \(object, x.new, lambda = 'lambda.min', gamma = 'gamma.min', ...) {

  if (!is.null(lambda)) if (lambda == 'lambda.min') lambda <- object$lambda.min
  if (!is.null(gamma)) if (gamma == 'gamma.min') gamma <- object$gamma.min
  predict.grpsel(object$fit, x.new, lambda = lambda, gamma = gamma, ...)

}

#==================================================================================================#
# Plot function
#==================================================================================================#

#' @title Plot function for cv.grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Plot the cross-validation results from group subset selection for a specified value
#' of \code{gamma}.
#'
#' @param x an object of class \code{cv.grpsel}
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A plot of the cross-validation results.
#'
#' @method plot cv.grpsel
#'
#' @export
#'
#' @importFrom graphics "plot"

plot.cv.grpsel <- \(x, gamma = 'gamma.min', ...) {

  if (gamma == 'gamma.min') gamma <- x$gamma.min
  index <- which.min(abs(gamma - x$gamma))
  df <- data.frame(cv.mean = x$cv.mean[[index]], cv.sd = x$cv.sd[[index]], ng = x$fit$ng[[index]])
  p <- ggplot2::ggplot(df, ggplot2::aes_string('ng', 'cv.mean')) +
    ggplot2::geom_point() +
    ggplot2::geom_errorbar(ggplot2::aes_string(ymin = 'cv.mean - cv.sd',
                                               ymax = 'cv.mean + cv.sd')) +
    ggplot2::xlab('number of groups')
  p

}
