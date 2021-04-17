#include <RcppArmadillo.h>
#include "cd.h"
#include "threshold.h"

// Triage update type by loss function

void cd::update(fit& fit, par& par, const arma::uvec& indices) {
  if (par.loss_fun == 1) cd::update_square(fit, par, indices);
  else cd::update_logistic(fit, par, indices);
}

// Coordinate descent update for square loss

void cd::update_square(fit& fit, par& par, const arma::uvec& indices) {

  for (arma::uword k : indices) {

    // Set group specific parameters

    bool isactive;
    double lambda = par.lambda * par.pen_fact(k, 0);
    double gamma1 = par.gamma1 * par.pen_fact(k, 1);
    double gamma2 = par.gamma2 * par.pen_fact(k, 2);
    double c = par.lips_const(k);
    arma::uvec group = par.groups(k);

    // Compute new estimate

    for (arma::uword j : group) fit.grad(j) = - arma::dot(fit.r, fit.x.unsafe_col(j));
    arma::vec beta = fit.beta(group);
    arma::vec beta_update = beta - fit.grad(group) / c;
    threshold(beta_update, isactive, lambda, gamma1, gamma2, c);

    if (isactive | fit.active(k)) {

      // Update residuals and coefficients

      arma::vec delta = beta_update - beta;
      fit.r -= fit.x.cols(group) * delta;
      fit.beta(group) = beta_update;

      double delta_infnorm = arma::norm(delta, "inf");
      if (delta_infnorm > max_delta) max_delta = delta_infnorm;
      double beta_infnorm = arma::norm(beta, "inf");
      if (beta_infnorm > max_beta) max_beta = beta_infnorm;

    }

    fit.active(k) = isactive;

  }

  iter++;

}

// Coordinate descent update for logistic loss

void cd::update_logistic(fit& fit, par& par, const arma::uvec& indices) {

  // Update intercept

  double intercept = fit.intercept;
  fit.intercept += arma::mean(fit.r) / 0.25;
  double delta = fit.intercept - intercept;
  fit.exb *= std::exp(- delta);
  fit.r = fit.y - 1 / (1 + fit.exb);
  max_delta = std::abs(delta);

  for (arma::uword k : indices) {

    // Set group specific parameters

    bool isactive;
    double lambda = par.lambda * par.pen_fact(k, 0);
    double gamma1 = par.gamma1 * par.pen_fact(k, 1);
    double gamma2 = par.gamma2 * par.pen_fact(k, 2);
    double c = par.lips_const(k);
    arma::uvec group = par.groups(k);

    // Compute new estimate

    for (arma::uword j : group) fit.grad(j) = - arma::dot(fit.r, fit.x.unsafe_col(j));
    arma::vec beta = fit.beta(group);
    arma::vec beta_update = beta - fit.grad(group) / c;
    threshold(beta_update, isactive, lambda, gamma1, gamma2, c);

    if (isactive | fit.active(k)) {

      // Update residuals and coefficients

      arma::vec delta = beta_update - beta;
      fit.exb %= arma::exp(- fit.x.cols(group) * delta);
      fit.r = fit.y - 1 / (1 + fit.exb);
      fit.beta(group) = beta_update;

      double delta_infnorm = arma::norm(delta, "inf");
      if (delta_infnorm > max_delta) max_delta = delta_infnorm;
      double beta_infnorm = arma::norm(beta, "inf");
      if (beta_infnorm > max_beta) max_beta = beta_infnorm;

    }

    fit.active(k) = isactive;

  }

  iter++;

}

// Coordinate descent algorithm

void cd::run(fit& fit, par& par) {

  // Compute gradient norms

  arma::uvec inactive_ind = arma::find(1 - fit.active);
  arma::vec grad_norm = arma::vec(fit.g, arma::fill::ones) * arma::datum::inf;
  arma::vec grad_norm_avg = grad_norm;
  if (sort | (par.lambda < 0)) {
    for (arma::uword k : inactive_ind) {
      arma::uvec group = par.groups(k);
      grad_norm(k) = arma::norm(fit.grad(group), 2);
      grad_norm_avg(k) = grad_norm(k) / std::sqrt(group.size());
    }
  }

  // Compute lambda

  if (par.lambda < 0) {
    par.lambda = 0;
    double lambda_tmp;
    for (arma::uword k : inactive_ind) {
      lambda_tmp = std::pow(std::max<double>(grad_norm(k) - par.gamma1 * par.pen_fact(k, 1), 0), 2) /
        (2 * std::max(par.pen_fact(k, 0), 1e-8) *
          (par.lips_const(k) + 2 * par.gamma2 * par.pen_fact(k, 2)));
      if (lambda_tmp > par.lambda) par.lambda = lambda_tmp;
    }
  par.lambda = par.alpha * par.lambda;
  }

  // Sort coordinates

  arma::uvec order;
  if (sort) order = arma::stable_sort_index(grad_norm_avg, "descend");
  else order = arma::linspace<arma::uvec>(0, fit.g - 1, fit.g);

  // Screen coordinates

  arma::uvec strong;
  arma::uvec weak;
  unsigned ng = sum(fit.active);
  bool doscreen = sort & (ng + screen < fit.g);
  if (doscreen) {
    strong = order.rows(0, ng + screen - 1);
    weak = order.rows(ng + screen, fit.g - 1);
  } else {
    strong = order;
  }

  // Run updates

  unsigned stable_count = 0;

  while (iter < max_iter) {

    max_delta = 0;
    max_beta = 1;
    arma::uvec active_old = fit.active;

    // Update strong set

    update(fit, par, strong);

    // Check convergence on strong set

    if (max_delta < eps * max_beta) {

      // If converged, check weak set for violations

      if (doscreen) {

        active_old = fit.active;
        update(fit, par, weak);
        arma::uvec violate = fit.active - active_old;

        if (arma::any(violate)) {

          // Move any violations from weak set to strong set

          weak.shed_rows(arma::find(violate.rows(weak)));
          strong.insert_rows(0, arma::find(violate));

        } else {

          break;

        }

      } else {

        break;

      }

    } else {

      // If not converged, check active set for stabilisation

      if (active_set) {

        if (arma::any(fit.active != active_old)) stable_count = 0;
        else stable_count++;

        if (stable_count == active_set_count - 1) {

          arma::uvec active_ind = arma::find(fit.active);

          while (iter < max_iter) {

            max_delta = 0;
            max_beta = 1;

            // Update active set

            update(fit, par, active_ind);

            // Check convergence on active set

            if (max_delta < eps * max_beta) break;

          }

        }

      }

    }

  }

}
