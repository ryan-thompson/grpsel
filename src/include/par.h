#ifndef par_H
#define par_H

// par class contains model parameters

struct par {

  const arma::field<arma::uvec> groups, groups_ind;
  const double lambda_step;
  double lambda;
  const double gamma1, gamma2;
  arma::mat pen_fact;
  const arma::vec lips_const;
  const unsigned loss_fun;
  bool orthogonal;

  par(const arma::field<arma::uvec>& groups, const arma::field<arma::uvec>& groups_ind,
      const double& lambda_step, double& lambda, const double& gamma1, const double& gamma2,
      const arma::mat& pen_fact, const arma::vec& lips_const, const unsigned loss_fun) :
      groups(groups), groups_ind(groups_ind), lambda_step(lambda_step), lambda(lambda),
      gamma1(gamma1), gamma2(gamma2), pen_fact(pen_fact), lips_const(lips_const),
      loss_fun(loss_fun) {

    if (loss_fun == 1) orthogonal = arma::all(lips_const == 1);

  }

};

#endif
