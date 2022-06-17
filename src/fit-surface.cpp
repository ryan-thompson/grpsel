#include <RcppArmadillo.h>
#include "cd.h"
#include "ls.h"
#include "surface.h"

// Provides an R interface for fitting the regularisation surface

// [[Rcpp::export]]
Rcpp::List fitsurface(const arma::mat& x, const arma::vec&y, const arma::field<arma::uvec>& groups,
                      const arma::field<arma::uvec>& groups_ind, const bool& run_ls,
                      const arma::mat& pen_fact, arma::field<arma::vec> lambda,
                      const arma::vec& gamma, const unsigned& shrinkage, const double& lambda_step,
                      const arma::uword& pmax, const arma::uword& gmax, const bool& active_set,
                      const unsigned &active_set_count, const bool& sort, const unsigned& screen,
                      const double& eps, const unsigned& max_cd_iter, const unsigned& max_ls_iter,
                      const arma::vec& lips_const, const unsigned& loss_fun) {

  cd cd(active_set, active_set_count, sort, screen, eps, max_cd_iter);
  ls ls(run_ls, max_ls_iter);
  surface surface(x, y, groups, groups_ind, pen_fact, lambda, gamma, shrinkage, lambda_step, pmax, gmax,
                  lips_const, loss_fun);
  surface.run(cd, ls);

  return Rcpp::List::create(Rcpp::Named("intercept") = surface.intercept,
                            Rcpp::Named("beta") = surface.beta,
                            Rcpp::Named("gamma") = surface.gamma,
                            Rcpp::Named("lambda") = surface.lambda,
                            Rcpp::Named("np") = surface.np,
                            Rcpp::Named("ng") = surface.ng,
                            Rcpp::Named("iter.cd") = surface.iter_cd,
                            Rcpp::Named("iter.ls") = surface.iter_ls,
                            Rcpp::Named("loss") = surface.loss);

}
