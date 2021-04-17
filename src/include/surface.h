#ifndef surface_H
#define surface_H
#include "cd.h"
#include "ls.h"

// surface contains data and functions for fitting regularisation surface

class surface {

public:
  const arma::mat x;
  const arma::vec y;
  const arma::field<arma::uvec> groups;
  const arma::mat pen_fact;
  arma::field<arma::vec> lambda;
  const arma::vec gamma;
  const unsigned shrinkage;
  const double alpha;
  const arma::uword pmax, gmax;
  const arma::vec lips_const;
  const unsigned loss_fun;
  unsigned p, g;
  arma::vec exb, r, grad;
  double int0 = 0;
  double null_dev = 0;
  arma::uword ngamma;
  arma::uvec nlambda;
  arma::field<arma::mat> beta;
  arma::field<arma::vec> intercept, loss;
  arma::field<arma::uvec> np, ng, iter_cd, iter_ls;

  surface(const arma::mat& x, const arma::vec& y, const arma::field<arma::uvec>& groups,
          const arma::mat& pen_fact, arma::field<arma::vec>& lambda, const arma::vec& gamma,
          const unsigned& shrinkage, const double& alpha, const arma::uword& pmax,
          const arma::uword& gmax, const arma::vec& lips_const, const unsigned& loss_fun) :
          x(x), y(y), groups(groups), pen_fact(pen_fact), lambda(lambda), gamma(gamma),
          shrinkage(shrinkage), alpha(alpha), pmax(pmax), gmax(gmax), lips_const(lips_const),
          loss_fun(loss_fun) {
    p = x.n_cols;
    g = groups.size();
    r = y;
    if (loss_fun == 2) {
      double ybar = arma::mean(y);
      int0 = std::log(ybar / (1 - ybar));
      exb = arma::vec(y.size(), arma::fill::ones) * std::exp(- int0);
      arma::vec pi = 1 / (1 + exb);
      r -= pi;
      null_dev -= arma::dot(y, arma::log(pi)) + arma::dot(1 - y,  arma::log(1 - pi));
    }
    grad = arma::vec(p);
    // grad = (r.t() * x).t();
    for (arma::uword j = 0; j < p; j++) grad(j) = - arma::dot(r, x.unsafe_col(j));
    ngamma = gamma.size();
    nlambda = arma::uvec(ngamma, arma::fill::zeros);
    intercept = arma::field<arma::vec>(ngamma);
    beta = arma::field<arma::mat>(ngamma);
    np = arma::field<arma::uvec>(ngamma);
    ng = arma::field<arma::uvec>(ngamma);
    iter_cd = arma::field<arma::uvec>(ngamma);
    iter_ls = arma::field<arma::uvec>(ngamma);
    loss = arma::field<arma::vec>(ngamma);
    for (arma::uword i = 0; i < ngamma; i++) {
      unsigned nlambda_i = lambda(i).size();
      nlambda(i) = nlambda_i;
      intercept(i) = arma::vec(nlambda_i, arma::fill::zeros);
      beta(i) = arma::mat(p, nlambda_i, arma::fill::zeros);
      np(i) = arma::uvec(nlambda_i, arma::fill::zeros);
      ng(i) = arma::uvec(nlambda_i, arma::fill::zeros);
      iter_cd(i) = arma::uvec(nlambda_i, arma::fill::zeros);
      iter_ls(i) = arma::uvec(nlambda_i, arma::fill::zeros);
      loss(i) = arma::vec(nlambda_i, arma::fill::zeros);
    }

  };

  void run(cd& cd, ls& ls);

};

#endif
