#include <RcppArmadillo.h>
#include "loss.h"
#include "fit.h"
#include "par.h"

double loss_(const fit& fit, const par& par) {
  double loss = 0;
  if (par.loss_fun == 1) {
    loss += 0.5 * arma::dot(fit.r, fit.r);
  } else {
    arma::vec pi = arma::clamp(1 / (1 + fit.exb), 1e-5, 1 - 1e-5);
    loss -= arma::dot(fit.y, arma::log(pi)) + arma::dot(1 - fit.y,  arma::log(1 - pi));
  }
  return loss;
}
