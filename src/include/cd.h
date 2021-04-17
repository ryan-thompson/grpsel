#ifndef cd_H
#define cd_H
#include "fit.h"
#include "par.h"

// cd class contains data and functions for coordinate descent algorithm

class cd {

public:

  unsigned iter = 0;
  double max_beta;

  cd(const bool& active_set, const unsigned& active_set_count, const bool& sort,
     const unsigned& screen, const double& eps, const unsigned& max_iter) : active_set(active_set),
     active_set_count(active_set_count), sort(sort), screen(screen), eps(eps), max_iter(max_iter)
    {};

  void run(fit& fit, par& par);

private:

  const bool active_set;
  const unsigned active_set_count;
  const bool sort;
  const unsigned screen;
  const double eps;
  const unsigned max_iter;
  double max_delta;

  void update(fit& fit, par& par, const arma::uvec& indices);
  void update_square(fit& fit, par& par, const arma::uvec& indices);
  void update_logistic(fit& fit, par& par, const arma::uvec& indices);

};

#endif
