#' @title Group subset selection
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Fits the regularisation surface for a regression model with a group subset selection
#' penalty. The group subset penalty can be combined with either a group lasso or ridge penalty
#' for shrinkage. The group subset parameter is \code{lambda} and the group lasso/ridge parameter is
#' \code{gamma}.
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
#' @param ls a logical indicating whether to perform local search after coordinate descent;
#' typically leads to higher quality solutions
#' @param nlambda the number of group subset regularisation parameters to evaluate when
#' \code{lambda} is computed automatically; may evaluate fewer parameters if \code{pmax} or
#' \code{gmax} is reached first
#' @param ngamma the number of group lasso or ridge regularisation parameters to evaluate when
#' \code{gamma} is computed automatically
#' @param gamma.max the maximum value for \code{gamma} when \code{penalty='grSubset+Ridge'}; when
#' \code{penalty='grSubset+grLasso'} \code{gamma.max} is computed automatically from the data
#' @param gamma.min the minimum value for \code{gamma} when \code{penalty='grSubset+Ridge'} and the
#' minimum value for \code{gamma} as a fraction of \code{gamma.max} when
#' \code{penalty='grSubset+grLasso'}
#' @param lambda an optional list of decreasing sequences of group subset parameters; the list
#' should contain a vector for each value of \code{gamma}
#' @param gamma an optional decreasing sequence of group lasso or ridge parameters
#' @param pmax the maximum number of predictors ever allowed to be active; ignored if \code{lambda}
#' is supplied
#' @param gmax the maximum number of groups ever allowed to be active; ignored if \code{lambda} is
#' supplied
#' @param subset.factor a vector of penalty factors applied to the group subset penalty; equal to
#' the group sizes by default
#' @param lasso.factor a vector of penalty factors applied to the group lasso penalty; equal to the
#' square root of the group sizes by default
#' @param ridge.factor a vector of penalty factors applied to the ridge penalty; equal to a
#' vector of ones by default
#' @param alpha the step size taken when computing \code{lambda} from the data; should be a value
#' strictly between 0 and 1; larger values typically lead to a finer grid of subset sizes
#' @param eps the convergence tolerance; convergence is declared when the relative maximum
#' difference in consecutive coefficients is less than \code{eps}
#' @param max.cd.iter the maximum number of coordinate descent iterations allowed per value of
#' \code{lambda} and \code{gamma}
#' @param max.ls.iter the maximum number of local search iterations allowed per value of
#' \code{lambda} and \code{gamma}
#' @param active.set a logical indicating whether to use active set updates; typically lowers the
#' run time
#' @param active.set.count the number of consecutive coordinate descent iterations in which a
#' subset should appear before running active set updates
#' @param sort a logical indicating whether to sort the coordinates before running coordinate
#' descent; required for gradient screening; typically leads to higher quality solutions
#' @param screen the number of groups to keep after gradient screening; smaller values typically
#' lower the run time
#' @param orthogonalise a logical indicating whether to orthogonalise within groups
#' @param warn a logical indicating whether to print a warning if the algorithms fail to converge
#'
#' @details For linear regression (loss='square') the response and predictors are centred about
#' zero and scaled to unit l2-norm. For logistic regression (loss='logistic') only the predictors
#' are centred and scaled and an intercept is fit during the course of the algorithm.
#'
#' @return An object of class \code{grpsel}; a list with the following components:
#' \item{beta}{a list of matrices whose columns contain fitted coefficients for a given value of
#' \code{lambda}; an individual matrix in the list for each value of \code{gamma}}
#' \item{gamma}{a vector containing the values of \code{gamma} used in the fit}
#' \item{lambda}{a list of vectors containing the values of \code{lambda} used in the fit; an
#' individual vector in the list for each value of \code{gamma}}
#' \item{np}{a list of vectors containing the number of active predictors per value of
#' \code{lambda}; an individual vector in the list for each value of \code{gamma}}
#' \item{ng}{a list of vectors containing the the number of active groups per value of
#' \code{lambda}; an individual vector in the list for each value of \code{gamma}}
#' \item{iter.cd}{a list of vectors containing the number of coordinate descent iterations per value
#' of \code{lambda}; an individual vector in the list for each value of \code{gamma}}
#' \item{iter.ls}{a list of vectors containing the number of local search iterations per value
#' of \code{lambda}; an individual vector in the list for each value of \code{gamma}}
#' \item{loss}{a list of vectors containing the evaluated loss function per value of \code{lambda}
#' evaluated; an individual vector in the list for each value of \code{gamma}}
#'
#' @references Thompson, R. and Vahid, F. (2021). 'Group selection and shrinkage with application to
#' sparse semiparametric modeling'. arXiv: \href{https://arxiv.org/abs/2105.12081}{2105.12081}.
#'
#' @example R/examples/example-grpsel.R
#'
#' @export

grpsel <- \(x, y, group = seq_len(ncol(x)),
            penalty = c('grSubset', 'grSubset+grLasso', 'grSubset+Ridge'),
            loss = c('square', 'logistic'), ls = FALSE, nlambda = 100, ngamma = 10, gamma.max = 100,
            gamma.min = 1e-4, lambda = NULL, gamma = NULL, pmax = ncol(x),
            gmax = length(unique(group)), subset.factor = NULL, lasso.factor = NULL,
            ridge.factor = NULL, alpha = 0.99, eps = 1e-4, max.cd.iter = 1e4, max.ls.iter = 100,
            active.set = TRUE, active.set.count = 3, sort = TRUE, screen = 500,
            orthogonalise = TRUE, warn = TRUE) {

  penalty <- match.arg(penalty)
  loss <- match.arg(loss)

  # Check data is valid
  if (!is.matrix(x)) x <- as.matrix(x)
  if (!is.matrix(y)) y <- as.matrix(y)
  if (anyNA(y)) stop('y contains NAs; remove or impute rows with missing values')
  if (anyNA(x)) stop('x contains NAs; remove or impute rows with missing values')
  attributes(y) <- list(dim = attributes(y)$dim)
  attributes(x) <- list(dim = attributes(x)$dim)
  if (loss == 'logistic' & !all(y %in% c(0, 1))) stop('y must take values in {0,1}')

  # Check arguments are valid
  if (nrow(x) != nrow(y)) stop('x and y must have the same number of observations')
  group.list <- is.list(group)
  if (group.list & length(unique(unlist(group))) != ncol(x)) {
    stop('each column of x must be assigned to a group')
  }
  if (!group.list & length(group) != ncol(x)) {
    stop('each column of x must be assigned to a group')
  }
  if (nlambda < 1) stop('nlambda must be at least one')
  if (ngamma < 1) stop('ngamma must be at least one')
  if (gamma.max <= 0) stop('gamma.max must be positive')
  if (gamma.min <= 0) stop('gamma.min must be positive')
  if (penalty != 'grSubset' & !is.null(lambda)) {
    if (is.null(gamma) & length(lambda) != ngamma) {
      stop('lambda must be a list with length ngamma')
    }
    if (!is.null(gamma) & length(lambda) != length(gamma)) {
      stop('lambda must be a list with the same length as gamma')
    }
  }
  if (group.list) g <- length(group) else g <- length(unique(group))
  if (!is.null(subset.factor) & length(subset.factor) != g) {
    stop('length of subset.factor must equal number of groups')
  }
  if (!is.null(lasso.factor) & length(lasso.factor) != g) {
    stop('length of lasso.factor must equal number of groups')
  }
  if (!is.null(ridge.factor) & length(ridge.factor) != g) {
    stop('length of ridge.factor must equal number of groups')
  }
  if (alpha >= 1 | alpha <= 0) stop('alpha must be between 0 and 1 (strictly)')

  # Expand predictor matrix if groups overlap
  if (group.list) {
    coef.id <- sort(unlist(group))
    x <- x[, coef.id]
    group <- rep(seq_along(group), times = lengths(group))[order(unlist(group))]
  }

  # Preliminaries
  n <- nrow(x)
  p <- ncol(x)
  pmax <- min(pmax, n - 1, p)

  # Standardise data
  x <- standardise(x)
  if (loss == 'square') y <- standardise(y)
  x.c <- attributes(x)$`scaled:center`
  x.s <- attributes(x)$`scaled:scale`
  y.c <- attributes(y)$`scaled:center`
  y.s <- attributes(y)$`scaled:scale`

  # Set up groups
  groups <- stats::aggregate(1:p ~ group, FUN = c, simplify = FALSE)[, 2]
  groups0 <- lapply(groups, '-', 1) # For C++
  gsize <- vapply(groups, length, integer(1))
  if (is.null(subset.factor)) subset.factor <- gsize
  if (is.null(lasso.factor)) lasso.factor <- sqrt(gsize)
  if (is.null(ridge.factor)) ridge.factor <- rep(1, g)
  penalty.factor <- cbind(subset.factor, lasso.factor, ridge.factor)

  # Orthogonalise groups
  if (orthogonalise) {
    x <- orthogonalise(x, groups0)
    z <- x$z
    x <- x$x
  }

  # Compute the Lipschitz constants
  if (orthogonalise | all(gsize == 1)) {
    lips.const <- rep(1, g)
  } else {
    lips.const <- lipschitz(x, groups0)
  }
  if (loss == 'logistic') lips.const <- lips.const / 4

  # Set up regularisation sequences
  if (is.null(gamma)) {
    if (penalty == 'grSubset') {
      ngamma <- 1
      gamma <- 0
    } else if (penalty == 'grSubset+grLasso') {
      ind <- unlist(groups[which(lasso.factor == 0)])
      if (is.null(ind)) {
        if (loss == 'square') {
          r <- y
        } else if (loss == 'logistic') {
          r <- stats::residuals(stats::glm(y ~ 1, family = 'binomial'), type = 'response')
        }
      } else {
        if (loss == 'square') {
          r <- stats::residuals(stats::glm(y ~ x[, ind], family = 'gaussian'), type = 'response')
        } else if (loss == 'logistic') {
          r <- stats::residuals(stats::glm(y ~ x[, ind], family = 'binomial'), type = 'response')
        }
      }
      xr <- crossprod(x, r)
      gamma.max <- max(vapply(which(lasso.factor != 0), \(l) sqrt(sum(xr[groups[[l]]] ^ 2)) /
                                lasso.factor[l], numeric(1)))
      gamma <- exp(seq(log(gamma.max), log(gamma.max * gamma.min), length.out = ngamma))
      gamma[1] <- gamma[1] * 1.00001 # Ensures first solution is zero when lambda=0
    } else if (penalty == 'grSubset+Ridge') {
      gamma <- exp(seq(log(gamma.max), log(gamma.min), length.out = ngamma))
    }
  } else {
    ngamma <- length(gamma)
  }
  if (is.null(lambda)) lambda <- replicate(ngamma, rep(- 1, nlambda), simplify = FALSE)

  # Fit regularisation surface
  result <- fitsurface(x, y, groups0, ls, penalty.factor, lambda, gamma,
                       which(penalty == c('grSubset', 'grSubset+grLasso', 'grSubset+Ridge')),
                       alpha, pmax, gmax, active.set, active.set.count, sort, screen, eps,
                       max.cd.iter, max.ls.iter, lips.const,
                       which(loss == c('square', 'logistic')))

  # Unwind orthogonalisation
  if (orthogonalise) result$beta <- lapply(result$beta, unorthogonalise, groups = groups0, z = z)

  # Unwind standardisation
  result$beta <- mapply(unstandardise, beta = result$beta, intercept = result$intercept,
                        MoreArgs = list(x.c = x.c, x.s = x.s, y.c = y.c, y.s = y.s, loss = loss),
                        SIMPLIFY = FALSE)
  result$intercept <- NULL

  # Aggregate coefficients if groups overlap
  if (group.list) {
    result$beta <- lapply(result$beta,
                          \(beta) stats::aggregate(beta ~ c(0, coef.id), FUN = sum)[, - 1])
    result$beta <- lapply(result$beta, \(beta) {colnames(beta) <- NULL; as.matrix(beta)})
  }

  # Warn if maximum iterations reached
  if (warn & any(unlist(result$iter.cd) == max.cd.iter)) {
    warning('coordinate descent did not converge for at least one set of regularisation parameters')
  }
  if (warn & any(unlist(result$iter.ls) == max.ls.iter)) {
    warning('local search did not converge for at least one set of regularisation parameters')
  }

  # Return result
  class(result) <- 'grpsel'
  return(result)

}

#==================================================================================================#
# Coefficient function
#==================================================================================================#

#' @title Coefficient function for grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Extracts coefficients for specified values of the tuning parameters.
#'
#' @param object an object of class \code{grpsel}
#' @param lambda the value of \code{lambda} indexing the desired fit
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A matrix of coefficients.
#'
#' @method coef grpsel
#'
#' @export
#'
#' @importFrom stats "coef"

coef.grpsel <- \(object, lambda = NULL, gamma = NULL, ...) {

  if (is.null(gamma) & is.null(lambda)) {
    do.call(cbind, object$beta)
  } else if (!is.null(gamma) & !is.null(lambda)) {
    index1 <- which.min(abs(gamma - object$gamma))
    index2 <- which.min(abs(lambda - object$lambda[[index1]]))
    object$beta[[index1]][, index2, drop = FALSE]
  } else if (!is.null(gamma) & is.null(lambda)) {
    index <- which.min(abs(gamma - object$gamma))
    object$beta[[index]]
  } else if (is.null(gamma) & !is.null(lambda)) {
    index <- vapply(object$lambda, \(x) which.min(abs(lambda - x)), integer(1))
    vapply(seq_along(object$gamma), \(x) object$beta[[x]][, index[x], drop = FALSE],
           numeric(nrow(object$beta[[1]])))
  }

}

#==================================================================================================#
# Predict function
#==================================================================================================#

#' @title Predict function for grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Generate predictions for new data using specified values of the tuning parameters.
#'
#' @param object an object of class \code{grpsel}
#' @param x.new a matrix of new values for the predictors
#' @param lambda the value of \code{lambda} indexing the desired fit
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A matrix of predictions.
#'
#' @method predict grpsel
#'
#' @export
#'
#' @importFrom stats "predict"

predict.grpsel <- \(object, x.new, lambda = NULL, gamma = NULL, ...) {

  beta <- coef.grpsel(object, lambda, gamma, ...)
  if (!is.matrix(x.new)) x.new <- as.matrix(x.new)
  cbind(1, x.new) %*% beta

}

#==================================================================================================#
# Plot function
#==================================================================================================#

#' @title Plot function for grpsel object
#'
#' @author Ryan Thompson <ryan.thompson@monash.edu>
#'
#' @description Plot the coefficient profiles from group subset selection for a specified value of
#' \code{gamma}.
#'
#' @param x an object of class \code{grpsel}
#' @param gamma the value of \code{gamma} indexing the desired fit
#' @param ... any other arguments
#'
#' @return A plot of the coefficient profiles.
#'
#' @method plot grpsel
#'
#' @export
#'
#' @importFrom graphics "plot"

plot.grpsel <- \(x, gamma = 0, ...) {

  index <- which.min(abs(gamma - x$gamma))
  beta <- x$beta[[index]]
  beta <- beta[- 1, , drop = FALSE]
  df <- data.frame(beta = as.vector(beta), predictor = as.factor(seq_along(beta[, 1])),
                   ng = rep(x$ng[[index]], each = nrow(beta)))
  df <- df[which(df$beta != 0), ]
  p <- ggplot2::ggplot(df, ggplot2::aes_string('ng', 'beta', col = 'predictor')) +
    ggplot2::geom_point() +
    ggplot2::xlab('number of groups')
  p

}
