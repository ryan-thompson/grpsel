#include <RcppArmadillo.h>

// Compute centers from matrix

// [[Rcpp::export]]
arma::vec centers(arma::mat x) {
  unsigned p = x.n_cols;
  arma::vec centers(p);
  for (arma::uword j = 0; j < p; j++) centers(j) = arma::mean(x.col(j));
  return centers;
}

// Compute scales from matrix

// [[Rcpp::export]]
arma::vec scales(arma::mat x) {
  unsigned p = x.n_cols;
  arma::vec scales(p);
  for (arma::uword j = 0; j < p; j++) scales(j) = arma::norm(x.col(j), 2);
  return scales;
}

// Center matrix

// [[Rcpp::export]]
arma::mat decenter(arma::mat x, const arma::vec& center) {
  unsigned p = x.n_cols;
  for (arma::uword j = 0; j < p; j++) x.col(j) -= center(j);
  return x;
}

// Scale matrix

// [[Rcpp::export]]
arma::mat descale(arma::mat x, const arma::vec& scale) {
  unsigned p = x.n_cols;
  for (arma::uword j = 0; j < p; j++) x.col(j) /= scale(j);
  return x;
}
