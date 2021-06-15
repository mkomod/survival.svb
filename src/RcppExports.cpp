// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// fit_exp
Rcpp::List fit_exp(arma::vec T, arma::vec delta, arma::mat X, double lambda, double a_0, double b_0, double a_omega, double b_omega, arma::vec m, arma::vec s, arma::vec g, int maxiter, bool verbose);
RcppExport SEXP _survival_svb_fit_exp(SEXP TSEXP, SEXP deltaSEXP, SEXP XSEXP, SEXP lambdaSEXP, SEXP a_0SEXP, SEXP b_0SEXP, SEXP a_omegaSEXP, SEXP b_omegaSEXP, SEXP mSEXP, SEXP sSEXP, SEXP gSEXP, SEXP maxiterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type T(TSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_0(a_0SEXP);
    Rcpp::traits::input_parameter< double >::type b_0(b_0SEXP);
    Rcpp::traits::input_parameter< double >::type a_omega(a_omegaSEXP);
    Rcpp::traits::input_parameter< double >::type b_omega(b_omegaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type s(sSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_exp(T, delta, X, lambda, a_0, b_0, a_omega, b_omega, m, s, g, maxiter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// fit_partial
Rcpp::List fit_partial(arma::vec T, arma::vec delta, arma::mat X, double lambda, double a_0, double b_0, arma::vec m, arma::vec s, arma::vec g, int maxiter, double tol, bool verbose);
RcppExport SEXP _survival_svb_fit_partial(SEXP TSEXP, SEXP deltaSEXP, SEXP XSEXP, SEXP lambdaSEXP, SEXP a_0SEXP, SEXP b_0SEXP, SEXP mSEXP, SEXP sSEXP, SEXP gSEXP, SEXP maxiterSEXP, SEXP tolSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type T(TSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type delta(deltaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_0(a_0SEXP);
    Rcpp::traits::input_parameter< double >::type b_0(b_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type m(mSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type s(sSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type g(gSEXP);
    Rcpp::traits::input_parameter< int >::type maxiter(maxiterSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_partial(T, delta, X, lambda, a_0, b_0, m, s, g, maxiter, tol, verbose));
    return rcpp_result_gen;
END_RCPP
}
// construct_risk_set
Rcpp::List construct_risk_set(const arma::vec& T, const arma::vec& delta);
RcppExport SEXP _survival_svb_construct_risk_set(SEXP TSEXP, SEXP deltaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type T(TSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type delta(deltaSEXP);
    rcpp_result_gen = Rcpp::wrap(construct_risk_set(T, delta));
    return rcpp_result_gen;
END_RCPP
}
// log_likelihood
double log_likelihood(const arma::vec& b, const arma::mat& X, const std::vector<arma::uvec>& R, const arma::uvec F);
RcppExport SEXP _survival_svb_log_likelihood(SEXP bSEXP, SEXP XSEXP, SEXP RSEXP, SEXP FSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type b(bSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const std::vector<arma::uvec>& >::type R(RSEXP);
    Rcpp::traits::input_parameter< const arma::uvec >::type F(FSEXP);
    rcpp_result_gen = Rcpp::wrap(log_likelihood(b, X, R, F));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_survival_svb_fit_exp", (DL_FUNC) &_survival_svb_fit_exp, 13},
    {"_survival_svb_fit_partial", (DL_FUNC) &_survival_svb_fit_partial, 12},
    {"_survival_svb_construct_risk_set", (DL_FUNC) &_survival_svb_construct_risk_set, 2},
    {"_survival_svb_log_likelihood", (DL_FUNC) &_survival_svb_log_likelihood, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_survival_svb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
