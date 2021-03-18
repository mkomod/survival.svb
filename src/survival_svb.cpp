#include "RcppArmadillo.h"

#include "exponential.hpp"
#include "optimiser.hpp"
#include "utils.hpp"


// [[Rcpp::export]]
arma::vec
init_P(arma::mat X, arma::vec m, arma::vec s, arma::vec g)
{
    int n = X.n_rows;
    int p = X.n_cols;
    arma::vec P(n, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
	for (int j = 0; j < p; ++j) {
	    // Prod x = exp(log(Prod x)) = exp(Sum log x))
	    P(i) += log(g(j) * normal_mgf(X(i, j), m(j), s(j)) + 1 - g(j));
	}
    }
    return exp(P);
}


// [[Rcpp::export]]
arma::vec
rm_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P /= g * normal_mgf(x_j, m, s) + (1 - g);
}


// [[Rcpp::export]]
arma::vec
add_P(arma::vec P, arma::vec x_j, double m, double s, double g)
{
    return P %= g * normal_mgf(x_j, m, s) + (1 - g);
}


// [[Rcpp::export]]
Rcpp::List 
fit(arma::vec T, arma::vec delta, arma::mat X, double omega, double lambda,
	double a_0,  double b_0, arma::vec m, arma::vec s, arma::vec g, 
	int maxiter, bool verbose)
{
    int p = X.n_cols;
    int n = X.n_rows;
    int n_delta = sum(delta);
    arma::vec m_old = arma::vec(p, arma::fill::randu);
    arma::vec s_old = arma::vec(p, arma::fill::randu);
    arma::vec g_old = arma::vec(p, arma::fill::randu);
    
    // P.i := see Eq(18)
    arma::vec P = init_P(X, m, s, g);
    
    for (int iter = 0; iter < maxiter; ++iter) {

	m_old = m;  s_old = s;  g_old = g;

	for (int j = 0; j < p; ++j) {
	    arma::colvec x_j = X.col(j);
	    double mu = m(j);
	    double sig = s(j);
	    double gam = g(j);
	    
	    // remove g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = rm_P(P, x_j, mu, sig, gam);

	    // optimise mu_j, sigma_j, gam_j
	    m(j) = optimise_mu_exp(sig, omega, lambda, P, T, delta,
		    x_j, verbose);
	    s(j) = optimise_sigma_exp(m(j), omega, lambda, P, T, delta,
		    x_j, verbose);
	    g(j) = optimise_gamma_exp(m(j), s(j), lambda, omega, a_0, b_0, 
		    P, T, delta, x_j, verbose);

	    // add in g_j M(x_j, m_j, s_j) - (1 - g_j) from P_i
	    P = add_P(P, x_j, m(j), s(j), g(j));
	}

	if (sum(abs(m_old - m)) < 1e-6 && sum(abs(s_old - s)) < 1e-6) {
	    if (verbose)
		Rcpp::Rcout << "Converged in: " << iter << " iterations\n";
	    break;
	}
    }
    

    return Rcpp::List::create(
	    Rcpp::Named("mu") = m,
	    Rcpp::Named("sigma") = s,
	    Rcpp::Named("gamma") = g
    );
}


// [[Rcpp::export]]
double 
objective_fn_exp(double mu, double sigma, double omega, double lambda, 
    const arma::vec &P, const arma::vec &T, const arma::vec x_j, bool verbose) 
{
    double res = sum(omega * T % normal_mgf(x_j, mu, sigma) % P - mu*x_j) +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}

