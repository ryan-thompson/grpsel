#include <RcppArmadillo.h>
#include "ls.h"
#include "threshold.h"

// Triage by loss function

void ls::run(fit& fit, par& par, cd& cd) {
  if (par.loss_fun == 1) {
    if (par.orthogonal) {
      ls::update_square_orthogonal(fit, par, cd);
    } else {
      ls::update_square(fit, par, cd);
    }
  } else {
    ls::update_logistic(fit, par, cd);
  }
}

// Local search update for square loss with orthogonal groups

void ls::update_square_orthogonal(fit& fit, par& par, cd& cd) {

  while (iter < max_iter) {

    arma::uvec active_ind = arma::find(fit.active);
    arma::uvec inactive_ind = arma::find(1 - fit.active);
    bool improved = false;
    unsigned ninactive = inactive_ind.size();
    if (ninactive == 0) break;

    // Outer loop over active set

    for (arma::uword k : active_ind) {

      // Set group specific parameters

      double lambda_k = par.lambda * par.pen_fact(k, 0);
      if (lambda_k == 0) continue; // Skip to next group if l20 penalisation is zero
      double gamma1_k = par.gamma1 * par.pen_fact(k, 1);
      double gamma2_k = par.gamma2 * par.pen_fact(k, 2);
      arma::uvec group_k = par.groups(k);
      arma::vec beta_k = fit.beta(group_k);
      double beta_k_l2norm = arma::norm(beta_k, 2);
      double obj_k = 0.5 * arma::dot(fit.r, fit.r) +
        lambda_k + gamma1_k * beta_k_l2norm + gamma2_k * beta_k_l2norm * beta_k_l2norm;

      // Remove group k from the fit

      arma::vec r_mk = fit.r + fit.x.cols(group_k) * beta_k;

      double best_obj = obj_k;
      arma::uword best_s;
      arma::vec best_beta;
      arma::vec best_r;

      // Inner loop over inactive set

      for (arma::uword s : inactive_ind) {

        // Set group specific parameters

        double lambda = par.lambda * par.pen_fact(s, 0);
        double gamma1 = par.gamma1 * par.pen_fact(s, 1);
        double gamma2 = par.gamma2 * par.pen_fact(s, 2);
        arma::uvec group = par.groups(s);

        bool isactive;

        // Compute partial minimiser

        arma::vec beta = fit.x.cols(group).t() * r_mk;
        threshold(beta, isactive, lambda, gamma1, gamma2, 1);

        // If minimiser is active, check if it improves on incumbent minimiser

        if (isactive) {
          arma::vec r = r_mk - fit.x.cols(group) * beta;
          double beta_l2norm = arma::norm(beta, 2);
          double obj = 0.5 * arma::dot(r, r) +
            lambda + gamma1 * beta_l2norm + gamma2 * beta_l2norm * beta_l2norm;
          if (obj < best_obj) {
            best_obj = obj;
            best_beta = beta;
            best_s = s;
            best_r = r;
          }
        }

      }

      // Update solution if there was an improvement

      if (best_obj < obj_k) {
        fit.beta(group_k).zeros();
        fit.active(k) = false;
        fit.beta(par.groups(best_s)) = best_beta;
        fit.active(best_s) = true;
        fit.r = best_r;
        for (arma::uword j = 0; j < fit.p; j++) fit.grad(j) = - arma::dot(fit.r, fit.x.unsafe_col(j));
        cd.run(fit, par);
        improved = true;
        break;
      }

    }

    iter++;

    // Exit if no improvement

    if (!improved) break;

  }

}

// Local search update for square loss

void ls::update_square(fit& fit, par& par, cd& cd) {

  while (iter < max_iter) {

    arma::uvec active_ind = arma::find(fit.active);
    arma::uvec inactive_ind = arma::find(1 - fit.active);
    bool improved = false;
    unsigned ninactive = inactive_ind.size();
    if (ninactive == 0) break;

    // Outer loop over active set

    for (arma::uword k : active_ind) {

      // Set group specific parameters

      double lambda_k = par.lambda * par.pen_fact(k, 0);
      if (lambda_k == 0) continue; // Skip to next group if l20 penalisation is zero
      double gamma1_k = par.gamma1 * par.pen_fact(k, 1);
      double gamma2_k = par.gamma2 * par.pen_fact(k, 2);
      arma::uvec group_k = par.groups(k);
      arma::vec beta_k = fit.beta(group_k);
      double beta_k_l2norm = arma::norm(beta_k, 2);
      double obj_k = 0.5 * arma::dot(fit.r, fit.r) +
        lambda_k + gamma1_k * beta_k_l2norm + gamma2_k * beta_k_l2norm * beta_k_l2norm;

      // Remove group k from the fit

      arma::vec r_mk = fit.r + fit.x.cols(group_k) * beta_k;

      // Sort inactive groups by their gradients

      arma::vec grad_groups = arma::vec(ninactive);
      for (arma::uword i = 0; i < ninactive; i++) {
        arma::uword k = inactive_ind(i);
        arma::uvec group = par.groups(k);
        grad_groups(i) = arma::norm(fit.x.cols(group).t() * r_mk, '2') / std::sqrt(group.size());
      }
      arma::uvec order = arma::stable_sort_index(grad_groups, "descend");
      unsigned top_k = std::round(fit.g * top_k_prop);
      top_k = std::min(std::max(top_k, top_k_min), ninactive);
      arma::uvec best_inactive_ind = inactive_ind.rows(order.rows(0, top_k - 1));

      double best_obj = obj_k;
      arma::uword best_s;
      arma::vec best_beta;
      arma::vec best_r;

      // Inner loop over inactive set

      for (arma::uword s : best_inactive_ind) {

        // Set group specific parameters

        double lambda = par.lambda * par.pen_fact(s, 0);
        double gamma1 = par.gamma1 * par.pen_fact(s, 1);
        double gamma2 = par.gamma2 * par.pen_fact(s, 2);
        double c = par.lips_const(s);
        arma::uvec group = par.groups(s);

        bool isactive;
        unsigned iter = 0;
        arma::vec r = r_mk;
        arma::vec beta = arma::vec(group.size(), arma::fill::zeros);

        // Compute partial minimiser

        while (iter < 50) {
          iter++;
          arma::vec beta_old = beta;
          beta += fit.x.cols(group).t() * r / c;
          threshold(beta, isactive, lambda, gamma1, gamma2, c);
          if (!isactive) {
            break;
          } else {
            arma::vec delta = beta - beta_old;
            r -= fit.x.cols(group) * delta;
            if (arma::norm(delta, "inf") < eps * cd.max_beta) break;
          }
        }

        // If minimiser is active, check if it improves on incumbent minimiser

        if (isactive) {
          double beta_l2norm = arma::norm(beta, 2);
          double obj = 0.5 * arma::dot(r, r) +
            lambda + gamma1 * beta_l2norm + gamma2 * beta_l2norm * beta_l2norm;
          if (obj < best_obj) {
            best_obj = obj;
            best_beta = beta;
            best_s = s;
            best_r = r;
          }
        }

      }

      // Update solution if there was an improvement

      if (best_obj < obj_k) {
        fit.beta(group_k).zeros();
        fit.active(k) = false;
        fit.beta(par.groups(best_s)) = best_beta;
        fit.active(best_s) = true;
        fit.r = best_r;
        for (arma::uword j = 0; j < fit.p; j++) fit.grad(j) = - arma::dot(fit.r, fit.x.unsafe_col(j));
        cd.run(fit, par);
        improved = true;
        break;
      }

    }

    iter++;

    // Exit if no improvement

    if (!improved) break;

  }

}

// Local search update for logistic loss

void ls::update_logistic(fit& fit, par& par, cd& cd) {

  while (iter < max_iter) {

    arma::uvec active_ind = arma::find(fit.active);
    arma::uvec inactive_ind = arma::find(1 - fit.active);
    bool improved = false;
    unsigned ninactive = inactive_ind.size();
    if (ninactive == 0) break;

    // Outer loop over active set

    for (arma::uword k : active_ind) {

      // Set group specific parameters

      double lambda_k = par.lambda * par.pen_fact(k, 0);
      if (lambda_k == 0) continue; // Skip to next group if l20 penalisation is zero
      double gamma1_k = par.gamma1 * par.pen_fact(k, 1);
      double gamma2_k = par.gamma2 * par.pen_fact(k, 2);
      arma::uvec group_k = par.groups(k);
      arma::vec beta_k = fit.beta(group_k);
      double beta_k_l2norm = arma::norm(beta_k, 2);
      arma::vec pi_k = arma::clamp(1 / (1 + fit.exb), 1e-15, 1 - 1e-15);
      double obj_k = - arma::dot(fit.y, arma::log(pi_k)) -
        arma::dot(1 - fit.y,  arma::log(1 - pi_k)) +
        lambda_k + gamma1_k * beta_k_l2norm + gamma2_k * beta_k_l2norm * beta_k_l2norm;

      // Remove group k from the fit

      arma::vec exb_mk = fit.exb / arma::exp(- fit.x.cols(group_k) * beta_k);
      arma::vec r_mk = fit.y - 1 / (1 + exb_mk);

      // Sort inactive groups by their gradients

      arma::vec grad_groups = arma::vec(ninactive);
      for (arma::uword i = 0; i < ninactive; i++) {
        arma::uword k = inactive_ind(i);
        arma::uvec group = par.groups(k);
        grad_groups(i) = arma::norm(fit.x.cols(group).t() * r_mk, '2') / std::sqrt(group.size());
      }
      arma::uvec order = arma::stable_sort_index(grad_groups, "descend");
      unsigned top_k = std::round(fit.g * top_k_prop);
      top_k = std::min(std::max(top_k, top_k_min), ninactive);
      arma::uvec best_inactive_ind = inactive_ind.rows(order.rows(0, top_k - 1));

      double best_obj = obj_k;
      arma::uword best_s;
      arma::vec best_beta;
      arma::vec best_exb;
      arma::vec best_r;

      // Inner loop over inactive set

      for (arma::uword s : best_inactive_ind) {

        // Set group specific parameters

        double lambda = par.lambda * par.pen_fact(s, 0);
        double gamma1 = par.gamma1 * par.pen_fact(s, 1);
        double gamma2 = par.gamma2 * par.pen_fact(s, 2);
        double c = par.lips_const(s);
        arma::uvec group = par.groups(s);

        bool isactive;
        unsigned iter = 0;
        arma::vec r = r_mk;
        arma::vec exb = exb_mk;
        arma::vec beta = arma::vec(group.size(), arma::fill::zeros);

        // Compute partial minimiser

        while (iter < 50) {
          iter++;
          arma::vec beta_old = beta;
          beta += fit.x.cols(group).t() * r / c;
          threshold(beta, isactive, lambda, gamma1, gamma2, c);
          if (!isactive) {
            break;
          } else {
            arma::vec delta = beta - beta_old;
            exb %= arma::exp(- fit.x.cols(group) * delta);
            r = fit.y - 1 / (1 + exb);
            if (arma::norm(delta, "inf") < eps * cd.max_beta) break;
          }
        }

        // If minimiser is active, check if it improves on incumbent minimiser

        if (isactive) {
          arma::vec pi = arma::clamp(1 / (1 + exb), 1e-15, 1 - 1e-15);
          double beta_l2norm = arma::norm(beta, 2);
          double obj = - arma::dot(fit.y, arma::log(pi)) -
            arma::dot(1 - fit.y, arma::log(1 - pi)) +
            lambda + gamma1 * beta_l2norm + gamma2 * beta_l2norm * beta_l2norm;
          if (obj < best_obj) {
            best_obj = obj;
            best_beta = beta;
            best_s = s;
            best_exb = exb;
            best_r = r;
          }
        }

      }

      // Update solution if there was an improvement

      if (best_obj < obj_k) {
        fit.beta(group_k).zeros();
        fit.active(k) = false;
        fit.beta(par.groups(best_s)) = best_beta;
        fit.active(best_s) = true;
        fit.exb = best_exb;
        fit.r = best_r;
        for (arma::uword j = 0; j < fit.p; j++) fit.grad(j) = - arma::dot(fit.r, fit.x.unsafe_col(j));
        cd.run(fit, par);
        improved = true;
        break;
      }

    }

    iter++;

    // Exit if no improvement

    if (!improved) break;

  }

}
