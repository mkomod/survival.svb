// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// fit_exp
Rcpp::List fit_exp(arma::vec T, arma::vec delta, arma::mat X, double lambda, double a_0, double b_0, double a_omega, double b_omega, arma::vec m, arma::vec s, arma::vec g, int maxiter, bool verbose, int threads);
RcppExport SEXP _survival_svb_fit_exp(SEXP TSEXP, SEXP deltaSEXP, SEXP XSEXP, SEXP lambdaSEXP, SEXP a_0SEXP, SEXP b_0SEXP, SEXP a_omegaSEXP, SEXP b_omegaSEXP, SEXP mSEXP, SEXP sSEXP, SEXP gSEXP, SEXP maxiterSEXP, SEXP verboseSEXP, SEXP threadsSEXP) {
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
    Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_exp(T, delta, X, lambda, a_0, b_0, a_omega, b_omega, m, s, g, maxiter, verbose, threads));
    return rcpp_result_gen;
END_RCPP
}
// fit_partial
Rcpp::List fit_partial(arma::vec T, arma::vec delta, arma::mat X, double lambda, double a_0, double b_0, arma::vec m, arma::vec s, arma::vec g, int maxiter, double tol, bool verbose, int threads);
RcppExport SEXP _survival_svb_fit_partial(SEXP TSEXP, SEXP deltaSEXP, SEXP XSEXP, SEXP lambdaSEXP, SEXP a_0SEXP, SEXP b_0SEXP, SEXP mSEXP, SEXP sSEXP, SEXP gSEXP, SEXP maxiterSEXP, SEXP tolSEXP, SEXP verboseSEXP, SEXP threadsSEXP) {
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
    Rcpp::traits::input_parameter< int >::type threads(threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(fit_partial(T, delta, X, lambda, a_0, b_0, m, s, g, maxiter, tol, verbose, threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_survival_svb_fit_exp", (DL_FUNC) &_survival_svb_fit_exp, 14},
    {"_survival_svb_fit_partial", (DL_FUNC) &_survival_svb_fit_partial, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_survival_svb(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
