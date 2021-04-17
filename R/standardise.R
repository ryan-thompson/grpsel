# Standardarisation
standardise <- function(x, center = T, scale = T) {
  if (is.double(center)) {
    x.c <- center
    center <- T
  } else if (center) {
    x.c <- centers(x)
  }
  if (center) x <- decenter(x, x.c)
  if (is.double(scale)) {
    x.s <- scale
    scale <- T
  } else if (scale) {
    x.s <- scales(x)
    if (any(x.s == 0)) x.s[x.s == 0] <- 1 # Handle constant variables
  }
  if (scale) x <- descale(x, x.s)
  if (center) attributes(x)$`scaled:center` <- as.numeric(x.c)
  if (scale) attributes(x)$`scaled:scale` <- as.numeric(x.s)
  return(x)
}

# Undo standardisation
unstandardise <- function(beta, intercept, x.c, x.s, y.c, y.s, loss) {
  if (loss == 'square') {
    beta <- beta / x.s * y.s
    intercept <- y.c - x.c %*% beta
  } else if (loss == 'logistic') {
    beta <- beta / x.s
    intercept <- t(intercept) - x.c %*% beta
  }
  beta <- rbind(intercept, beta)
  return(beta)
}
