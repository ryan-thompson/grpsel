#ifndef ls_H
#define ls_H
#include "cd.h"

// ls class contains data and functions for local search algorithm

class ls {

public:

  bool run_ls;
  unsigned iter = 0;

  ls(const bool& run_ls, const unsigned& max_iter) : run_ls(run_ls), max_iter(max_iter) {};

  void run(fit& fit, par& par, cd& cd);

private:

  const unsigned max_iter;
  const unsigned top_k_min = 10;
  const double top_k_prop = 0.05;
  const double eps = 1e-2;

  void update_square_orthogonal(fit& fit, par& par, cd& cd);
  void update_square(fit& fit, par& par, cd& cd);
  void update_logistic(fit& fit, par& par, cd& cd);

};

#endif
