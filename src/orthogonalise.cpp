#include <RcppArmadillo.h>

// Orthogonalise groups

// [[Rcpp::export]]
Rcpp::List orthogonalise(arma::mat x, const arma::field<arma::uvec>& groups) {

  unsigned g = groups.size();
  arma::field<arma::mat> z(g, 1);

  for (arma::uword l = 0; l < g; l++) {

    arma::mat u, v;
    arma::vec s;
    arma::uvec group = groups(l);
    unsigned group_size = group.size();

    if (group_size == 1) {
      v = 1;
    } else {
      arma::svd_econ(u, s, v, x.cols(group), "right");
      for (arma::uword j = 0; j < group_size; j++) {
        if (s(j) == 0) s(j) = 1; // Handles constant columns
        v.col(j) /= s(j);
      }
      x.cols(group) = x.cols(group) * v;
    }
    z(l, 0) = v;

  }

  return Rcpp::List::create(Rcpp::Named("x") = x, Rcpp::Named("z") = z);

}

// Unorthogonalise groups

// [[Rcpp::export]]
arma::mat unorthogonalise(arma::mat beta, const arma::field<arma::uvec>& groups,
                          const arma::field<arma::mat>& z) {

  unsigned g = groups.size();

  for (arma::uword l = 0; l < g; l++) {
    arma::uvec group = groups(l);
    if (group.size() > 1) beta.rows(group) = z(l) * beta.rows(group);
  }

  return beta;

}
