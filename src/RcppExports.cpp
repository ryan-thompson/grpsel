// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// fitsurface
Rcpp::List fitsurface(const arma::mat& x, const arma::vec& y, const arma::field<arma::uvec>& groups, const bool& run_ls, const arma::mat& pen_fact, arma::field<arma::vec> lambda, const arma::vec& gamma, const unsigned& shrinkage, const double& alpha, const arma::uword& pmax, const arma::uword& gmax, const bool& active_set, const unsigned& active_set_count, const bool& sort, const unsigned& screen, const double& eps, const unsigned& max_cd_iter, const unsigned& max_ls_iter, const arma::vec& lips_const, const unsigned& loss_fun);
RcppExport SEXP _grpsel_fitsurface(SEXP xSEXP, SEXP ySEXP, SEXP groupsSEXP, SEXP run_lsSEXP, SEXP pen_factSEXP, SEXP lambdaSEXP, SEXP gammaSEXP, SEXP shrinkageSEXP, SEXP alphaSEXP, SEXP pmaxSEXP, SEXP gmaxSEXP, SEXP active_setSEXP, SEXP active_set_countSEXP, SEXP sortSEXP, SEXP screenSEXP, SEXP epsSEXP, SEXP max_cd_iterSEXP, SEXP max_ls_iterSEXP, SEXP lips_constSEXP, SEXP loss_funSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const bool& >::type run_ls(run_lsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type pen_fact(pen_factSEXP);
    Rcpp::traits::input_parameter< arma::field<arma::vec> >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type shrinkage(shrinkageSEXP);
    Rcpp::traits::input_parameter< const double& >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const arma::uword& >::type pmax(pmaxSEXP);
    Rcpp::traits::input_parameter< const arma::uword& >::type gmax(gmaxSEXP);
    Rcpp::traits::input_parameter< const bool& >::type active_set(active_setSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type active_set_count(active_set_countSEXP);
    Rcpp::traits::input_parameter< const bool& >::type sort(sortSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type screen(screenSEXP);
    Rcpp::traits::input_parameter< const double& >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type max_cd_iter(max_cd_iterSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type max_ls_iter(max_ls_iterSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lips_const(lips_constSEXP);
    Rcpp::traits::input_parameter< const unsigned& >::type loss_fun(loss_funSEXP);
    rcpp_result_gen = Rcpp::wrap(fitsurface(x, y, groups, run_ls, pen_fact, lambda, gamma, shrinkage, alpha, pmax, gmax, active_set, active_set_count, sort, screen, eps, max_cd_iter, max_ls_iter, lips_const, loss_fun));
    return rcpp_result_gen;
END_RCPP
}
// lipschitz
arma::vec lipschitz(arma::mat x, const arma::field<arma::uvec>& groups);
RcppExport SEXP _grpsel_lipschitz(SEXP xSEXP, SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(lipschitz(x, groups));
    return rcpp_result_gen;
END_RCPP
}
// orthogonalise
Rcpp::List orthogonalise(arma::mat x, const arma::field<arma::uvec>& groups);
RcppExport SEXP _grpsel_orthogonalise(SEXP xSEXP, SEXP groupsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type groups(groupsSEXP);
    rcpp_result_gen = Rcpp::wrap(orthogonalise(x, groups));
    return rcpp_result_gen;
END_RCPP
}
// unorthogonalise
arma::mat unorthogonalise(arma::mat beta, const arma::field<arma::uvec>& groups, const arma::field<arma::mat>& z);
RcppExport SEXP _grpsel_unorthogonalise(SEXP betaSEXP, SEXP groupsSEXP, SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::mat>& >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(unorthogonalise(beta, groups, z));
    return rcpp_result_gen;
END_RCPP
}
// centers
arma::vec centers(arma::mat x);
RcppExport SEXP _grpsel_centers(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(centers(x));
    return rcpp_result_gen;
END_RCPP
}
// scales
arma::vec scales(arma::mat x);
RcppExport SEXP _grpsel_scales(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(scales(x));
    return rcpp_result_gen;
END_RCPP
}
// decenter
arma::mat decenter(arma::mat x, const arma::vec& center);
RcppExport SEXP _grpsel_decenter(SEXP xSEXP, SEXP centerSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type center(centerSEXP);
    rcpp_result_gen = Rcpp::wrap(decenter(x, center));
    return rcpp_result_gen;
END_RCPP
}
// descale
arma::mat descale(arma::mat x, const arma::vec& scale);
RcppExport SEXP _grpsel_descale(SEXP xSEXP, SEXP scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type scale(scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(descale(x, scale));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_grpsel_fitsurface", (DL_FUNC) &_grpsel_fitsurface, 20},
    {"_grpsel_lipschitz", (DL_FUNC) &_grpsel_lipschitz, 2},
    {"_grpsel_orthogonalise", (DL_FUNC) &_grpsel_orthogonalise, 2},
    {"_grpsel_unorthogonalise", (DL_FUNC) &_grpsel_unorthogonalise, 3},
    {"_grpsel_centers", (DL_FUNC) &_grpsel_centers, 1},
    {"_grpsel_scales", (DL_FUNC) &_grpsel_scales, 1},
    {"_grpsel_decenter", (DL_FUNC) &_grpsel_decenter, 2},
    {"_grpsel_descale", (DL_FUNC) &_grpsel_descale, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_grpsel(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
