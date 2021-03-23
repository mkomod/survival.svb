#include "exponential.hpp"

struct kwargs {
    const double &mu;
    const double &sigma;
    const double &lambda;
    const double &omega;
    const arma::vec &P;
    const arma::vec &T; 
    const arma::vec &delta;
    const arma::vec &x_j;
    bool verbose;
};

struct exp_kwargs {
    const double &a;
    const double &b;
    const double &a_omega;
    const double &b_omega;
    const arma::vec &P;
    const arma::vec &T; 
    const int &n_delta;
    bool verbose;
};


static double 
f_exp_mu(double mu, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double sigma = arg->sigma;
    double omega = arg->omega;
    double lambda = arg->lambda;
    const arma::vec &P = arg->P;
    const arma::vec &T = arg->T;
    const arma::vec &delta = arg->delta;
    const arma::vec x_j = arg->x_j;
    bool verbose = arg->verbose;

    double res = sum(omega * T % normal_mgf(x_j, mu, sigma) % P 
	    - mu * delta % x_j) +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


static double 
f_exp_sig(double sigma, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double mu = arg->mu;
    double omega = arg->omega;
    double lambda = arg->lambda;
    const arma::vec &P = arg->P;
    const arma::vec &T = arg->T;
    const arma::vec &delta = arg->delta;
    const arma::vec x_j = arg->x_j;
    bool verbose = arg->verbose;

    double res = sum(omega * T % normal_mgf(x_j, mu, sigma) % P 
	    - mu * delta % x_j) +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


static double
f_exp_a(double a, void* args)
{
    exp_kwargs *arg = static_cast<exp_kwargs *>(args);
    double b = arg->b;
    double a_omega = arg->a_omega;
    double b_omega = arg->b_omega;
    const arma::vec &P = arg->P;
    const arma::vec &T = arg->T;
    const int n_delta = arg->n_delta;
    bool verbose = arg->verbose;

    double res = (sum(P % T) + b_omega) * (a / b) - a + 
	(a + a_omega - n_delta) * R::digamma(a) +
	// (n_delta - a_omega) * log(b) -
	R::lgammafn(a);
    
    if (verbose)
	Rcpp::Rcout << "a: " << res << "\n";

    return res;
};


double
opt_exp_gam(double mu, double sigma, double omega, double lambda, 
	double a_0, double b_0, const arma::vec &P, const arma::vec &T,
	const arma::vec &delta, const arma::vec &x_j, bool verbose) 
{
    double res = sigmoid(log(a_0 / b_0) + 1.0/2.0 -
	    (lambda * sigma * sqrt(2.0 / PI) * exp(-pow(mu/sigma, 2 )) +
	     lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) +
	     log(sqrt(2.0 / PI) * 1.0 /(sigma * lambda)) +
	     sum(omega * T % P % (normal_mgf(x_j, mu, sigma) - 1) - 
		 mu * delta % x_j)));

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


double
opt_exp_mu(double sigma, double omega, double lambda, 
    const arma::vec &P, const arma::vec &T, const arma::vec &delta,
    const arma::vec x_j, bool verbose)
{
    kwargs args = {0.0, sigma, lambda, omega, P, T, delta, x_j, verbose};
    return Brent_fmin(-1e2, 1e2, f_exp_mu, static_cast<void*>(&args), 1e-5);
}


double
opt_exp_sig(double mu, double omega, double lambda, 
    const arma::vec &P, const arma::vec &T, const arma::vec &delta,
    const arma::vec x_j, bool verbose)
{
    kwargs args = {mu, 0.0, lambda, omega, P, T, delta, x_j, verbose};
    return Brent_fmin(0, 4, f_exp_sig, static_cast<void*>(&args), 1e-5);
}


double 
opt_exp_a(double b, double a_omega, double b_omega,
    const arma::vec &P, const arma::vec &T, const int n_delta,
    bool verbose)
{
    exp_kwargs args = {0.0, b, a_omega, b_omega, P, T, n_delta, verbose};
    return Brent_fmin(0, 1e100, f_exp_a, static_cast<void *>(&args), 1e-5);
}


double 
opt_exp_b(double a, double a_omega, double b_omega, const arma::vec &P, 
	const arma::vec &T, int n_delta, bool verbose)
{
    return (b_omega + sum(P % T)) * a / (n_delta + a_omega);
}

