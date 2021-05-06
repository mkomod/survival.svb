#include <math.h>
#include <omp.h>
#include <vector>

#include "RcppArmadillo.h"

#include "exponential.hpp"
#include "optimiser.hpp"
#include "partial.hpp"
#include "utils.hpp"


// [[Rcpp::export]]
Rcpp::List 
fit_exp(arma::vec T, arma::vec delta, arma::mat X, double lambda,
	double a_0,  double b_0, double a_omega, double b_omega, 
	arma::vec m, arma::vec s, arma::vec g, int maxiter, bool verbose,
	int threads)
{
    omp_set_num_threads(threads);

    int p = X.n_cols;
    int n = X.n_rows;
    arma::vec m_old = arma::vec(p, arma::fill::zeros);
    arma::vec s_old = arma::vec(p, arma::fill::zeros);
    arma::vec g_old = arma::vec(p, arma::fill::zeros);
    double a_old, b_old, omega;
    double a = a_omega; double b = b_omega;
    int n_delta = sum(delta);
   
    arma::vec P = init_P(X, m, s, g);
    
    for (int iter = 0; iter < maxiter; ++iter) {

	m_old = m;  s_old = s;  g_old = g; a_old = a; b_old = b;

	a = opt_exp_a(b, a_omega, b_omega, P, T, n_delta);
	b = opt_exp_b(a, a_omega, b_omega, P, T, n_delta);

	for (int j = 0; j < p; ++j) {
	    arma::colvec x_j = X.col(j);
	    omega = a / b;
	    
	    // remove g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = rm_P(P, x_j, m(j), s(j), g(j));

	    m(j) = opt_exp_mu(s(j), omega, lambda, P, T, delta, x_j);
	    s(j) = opt_exp_sig(m(j), omega, lambda, P, T, delta, x_j);
	    g(j) = opt_exp_gam(m(j), s(j), omega, lambda, a_0, b_0, 
		    P, T, delta, x_j);

	    // add in g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = add_P(P, x_j, m(j), s(j), g(j));
	}

	if (sum(abs(m_old - m)) < 1e-6 && sum(abs(s_old - s)) < 1e-6 &&
	    std::abs(a_old - a) < 1e-6 && std::abs(b_old - b) < 1e-6) {
	    if (verbose)
		Rcpp::Rcout << "Converged in " << iter << " iterations\n";
	    break;
	}
    }
    

    return Rcpp::List::create(
	Rcpp::Named("mu") = m,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("a") = a,
	Rcpp::Named("b") = b
    );
}


// [[Rcpp::export]]
Rcpp::List
fit_partial(arma::vec T, arma::vec delta, arma::mat X, double lambda, 
	double a_0, double b_0, arma::vec m, arma::vec s, arma::vec g,
	int maxiter, double tol, bool verbose, int threads)
{
    omp_set_num_threads(threads);

    // indices of failure times
    arma::uvec F = find(delta);
    
    // construct list of indices between two failure times
    arma::vec FT = sort(T(F));
    std::vector<arma::uvec> R;
    for (int i = 0; i < FT.size() - 1; ++i)
	R.push_back(find(T < FT(i+1) && T >= FT(i)));
    R.push_back(find(T >= FT(FT.size() - 1)));

    int p = X.n_cols;
    arma::vec m_old, s_old, g_old;
    arma::vec P = init_log_P(X, m, s, g);

    for (int iter = 0; iter < maxiter; ++iter) {
	Rcpp::checkUserInterrupt();

	// init old
	m_old = m; s_old = s; g_old = g;

	for (int j = 0; j < p; ++j) {
	    arma::vec x_j = X.col(j);

	    P = rm_log_P(P, x_j, m(j), s(j), g(j));

	    m(j) = opt_par_mu(s(j), lambda, R, F, P, x_j);
	    s(j) = opt_par_sig(m(j), lambda, R, F, P, x_j);
	    g(j) = opt_par_gam(m(j), s(j), a_0, b_0, lambda, R, F, P, x_j);

	    P = add_log_P(P, x_j, m(j), s(j), g(j));
	    
	    // check for overflow
	    if (P.has_nan() || P.has_inf()) {
		Rcpp::Rcout << "\nOverflow error after updating parameter " << j + 1 << 
		    ".\n\n This may be a result of large values in X or large starting values.\n" <<
		    "  max(X[ , " << j + 1 << "]) = " << max(x_j) << 
		    "row num : " << index_max(x_j)+1 <<
		    ".\n\n Try rescaling X or using different starting values.\n\n";
		return Rcpp::List::create();
	    }

	}

	// check convergence
	if (sum(abs(m - m_old)) < tol && sum(abs(s - s_old)) < tol && sum(abs(g - g_old)) < tol) {
	    if (verbose)
		Rcpp::Rcout << "Converged in " << iter << " iterations\n";
	    return Rcpp::List::create(
		Rcpp::Named("mu") = m,
		Rcpp::Named("sigma") = s,
		Rcpp::Named("gamma") = g
	    );
	}
    }
    
    if (verbose)
	Rcpp::Rcout << "Failed to converge in " << maxiter << " iterations.\n";


    return Rcpp::List::create(
	Rcpp::Named("mu") = m,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g
    );
}


