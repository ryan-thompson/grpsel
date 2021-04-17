#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::vec lipschitz(arma::mat x, const arma::field<arma::uvec>& groups) {
  unsigned g = groups.size();
  arma::vec c(g);
  for (arma::uword l = 0; l < g; l++) c(l) = std::pow(arma::norm(x.cols(groups(l))), 2);
  return c;
}
