#include <RcppArmadillo.h>
#include "threshold.h"

// Threshold function

void threshold(arma::vec& beta, bool& isactive, const double& lambda, const double& gamma1,
                const double& gamma2, const double& c) {
  double beta_norm = arma::norm(beta, 2);
  double beta_update;
  double c2gamma2 = c + 2 * gamma2;
  double phi = c / c2gamma2 * (1 - gamma1 / (c * beta_norm));
  // if (phi * beta_norm >= std::sqrt(2 * lambda * (1 + 2 * gamma2)) / c2gamma2) {
  if (phi * beta_norm >= std::sqrt(2 * lambda / c2gamma2)) {
    beta_update = phi;
    isactive = beta_update != 0;
  } else {
    beta_update = 0;
    isactive = false;
  }
  beta *= beta_update;
}
