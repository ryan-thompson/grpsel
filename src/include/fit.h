#ifndef fit_H
#define fit_H

// fit struct contains model fit

struct fit {

  const arma::mat x;
  const arma::vec y;
  arma::vec r, grad, exb;
  double intercept;
  const unsigned p, g, sumpk;
  arma::vec beta = arma::vec(sumpk, arma::fill::zeros);
  arma::uvec active = arma::uvec(g, arma::fill::zeros);

  fit(const arma::mat& x, const arma::vec& y, arma::vec r, arma::vec grad, arma::vec exb,
      double intercept, const unsigned& p, const unsigned& g, const unsigned& sumpk) : x(x), y(y),
      r(r), grad(grad), exb(exb), intercept(intercept), p(p), g(g), sumpk(sumpk) {}

};

#endif
