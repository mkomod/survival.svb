#include <math.h>

#include "RcppArmadillo.h"

#include "exponential.hpp"
#include "partial.hpp"
#include "optimiser.hpp"
#include "utils.hpp"


// [[Rcpp::export]]
Rcpp::List 
fit_exp(arma::vec T, arma::vec delta, arma::mat X, double lambda,
	double a_0,  double b_0, double a_omega, double b_omega, 
	arma::vec m, arma::vec s, arma::vec g, int maxiter, bool verbose)
{
    int p = X.n_cols;
    int n = X.n_rows;
    arma::vec m_old = arma::vec(p, arma::fill::zeros);
    arma::vec s_old = arma::vec(p, arma::fill::zeros);
    arma::vec g_old = arma::vec(p, arma::fill::zeros);
    double a_old, b_old, omega;
    double a = a_omega; double b = b_omega;
    int n_delta = sum(delta);
   
    // 
    arma::vec P = init_P(X, m, s, g);
    
    for (int iter = 0; iter < maxiter; ++iter) {

	m_old = m;  s_old = s;  g_old = g; a_old = a; b_old = b;

	a = opt_exp_a(b, a_omega, b_omega, P, T, n_delta, verbose);
	b = opt_exp_b(a, a_omega, b_omega, P, T, n_delta, verbose);

	for (int j = 0; j < p; ++j) {
	    arma::colvec x_j = X.col(j);
	    omega = a / b;
	    
	    // remove g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = rm_P(P, x_j, m(j), s(j), g(j));

	    m(j) = opt_exp_mu(s(j), omega, lambda, P, T, delta, x_j, verbose);
	    s(j) = opt_exp_sig(m(j), omega, lambda, P, T, delta, x_j, verbose);
	    g(j) = opt_exp_gam(m(j), s(j), omega, lambda, a_0, b_0, 
		    P, T, delta, x_j, verbose);

	    // add in g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = add_P(P, x_j, m(j), s(j), g(j));
	}

	if (sum(abs(m_old - m)) < 1e-6 && sum(abs(s_old - s)) < 1e-6 &&
	    std::abs(a_old - a) < 1e-6 && std::abs(b_old - b) < 1e-6) {
	    if (verbose)
		Rcpp::Rcout << "Converged in: " << iter << " iterations\n";
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
	int maxiter, bool verbose)
{
    int p = X.n_cols;

    // indices of failed times from smallest to largest
    arma::uvec R = sort_index(T); 
    arma::uvec F = R(find(delta(R)));
    
    arma::vec m_old, s_old, g_old;
    arma::vec P = init_P(X, m, s, g);

    for (int iter = 0; iter < maxiter; ++iter) {
	// init old
	m_old = m; s_old = s; g_old = g;

	for (int j = 0; j < p; ++j) {
	    arma::vec x_j = X.col(j);

	    P = rm_P(P, x_j, m(j), s(j), g(j));

	    m(j) = opt_par_mu(s(j), lambda, T, F, P, x_j);
	    s(j) = opt_par_sig(m(j), lambda, T, F, P, x_j);
	    g(j) = opt_par_gam(m(j), s(j), a_0, b_0, lambda, T, F, P, x_j);

	    P = add_P(P, x_j, m(j), s(j), g(j));
	}

	// check convergence
	if (sum(abs(m - m_old)) < 1e-6 && sum(abs(s - s_old)) < 1e-6)
	    break;
    }


    return Rcpp::List::create(
	Rcpp::Named("mu") = m,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g
    );
}
