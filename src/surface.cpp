#include <RcppArmadillo.h>
#include "surface.h"
#include "cd.h"
#include "ls.h"
#include "fit.h"
#include "par.h"
#include "loss.h"

// Fit regularisation surface

void surface::run(cd& cd, ls& ls) {

  // Loop over gamma values

  for (arma::uword i = 0; i < ngamma; i++) {

    // Set current value of gamma; gamma1 is group lasso and gamma2 is ridge

    double gamma1 = 0;
    double gamma2 = 0;
    if (shrinkage == 2) gamma1 = gamma(i);
    else if (shrinkage == 3) gamma2 = gamma(i);

    // Loop over lambda values

    double lambda_step_j = 1.000001; // Ensures first solution contains only unpenalised groups
    bool compute_lambda = lambda(i)(0) < 0;
    arma::uword j;
    arma::uword nlambda_i = nlambda(i);
    fit fit(x, y, r, grad, exb, int0, p, g, sumpk);

    for (j = 0; j < nlambda_i; j++) {

      cd.iter = 0;
      ls.iter = 0;

      // Run coordinate descent for fixed values of lambda/gamma

      double lambda0 = lambda(i)(j);
      par par(groups, groups_ind, lambda_step_j, lambda0, gamma1, gamma2, pen_fact, lips_const,
              loss_fun);
      cd.run(fit, par);

      // // Exit if any NaNs (e.g., constant response or predictors)
      //
      // if (fit.beta.has_nan()) {
      //   if (j > 0) j--;
      //   break;
      // }

      // Run local search for fixed values of lambda/gamma

      if (ls.run_ls) ls.run(fit, par, cd);

      // Exit if pmax or gmax exceeded

      arma::uword nnz_p;
      if (fit.p == fit.sumpk) {
        nnz_p = arma::sum(fit.beta != 0);
      } else {
        arma::uvec nz = arma::uvec(fit.p, arma::fill::zeros);
        arma::uvec active_ind = arma::find(fit.active);
        for (arma::uword k : active_ind) for (arma::uword j : par.groups(k)) nz(j) = true;
        nnz_p = arma::sum(nz);
      }
      arma::uword nnz_g = arma::sum(fit.active);
      if (compute_lambda & ((nnz_p > pmax) | (nnz_g > gmax))) {
        j--;
        break;
      }

      // Save solution for current point in path

      intercept(i)(j) = fit.intercept;
      beta(i).col(j) = fit.beta;
      np(i)(j) = nnz_p;
      ng(i)(j) = nnz_g;
      lambda(i)(j) = par.lambda;
      iter_cd(i)(j) = cd.iter;
      iter_ls(i)(j) = ls.iter;
      loss(i)(j) = loss_(fit, par);

      lambda_step_j = lambda_step;

      // Exit if pmax or gmax reached

      if (compute_lambda & ((nnz_p == pmax) | (nnz_g == gmax))) break;

      // Exit if saturated (for logistic loss)

      if (compute_lambda & (par.loss_fun == 2) & (loss(i)(j) < 0.01 * null_dev)) break;

    }

    // Trim any empty fits

    if (j == nlambda_i) j--;
    if (j < nlambda_i - 1) {
      intercept(i).shed_rows(j + 1, nlambda_i - 1);
      beta(i).shed_cols(j + 1, nlambda_i - 1);
      lambda(i).shed_rows(j + 1, nlambda_i - 1);
      np(i).shed_rows(j + 1, nlambda_i - 1);
      ng(i).shed_rows(j + 1, nlambda_i - 1);
      iter_cd(i).shed_rows(j + 1, nlambda_i - 1);
      iter_ls(i).shed_rows(j + 1, nlambda_i - 1);
      loss(i).shed_rows(j + 1, nlambda_i - 1);
    }

  }

}
