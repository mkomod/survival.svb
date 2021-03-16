#include "exponential.hpp"

struct kwargs {
    const double &mu;
    const double &sigma;
    const double &lambda;
    const double &omega;
    const arma::vec &P;
    const arma::vec &T; 
    const arma::vec &x_j;
    bool verbose;
};


double 
objective_mu_exp(double mu, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double sigma = arg->sigma;
    double omega = arg->omega;
    double lambda = arg->lambda;
    const arma::vec &P = arg->P;
    const arma::vec &T = arg->T;
    const arma::vec x_j = arg->x_j;
    bool verbose = arg->verbose;

    double res = sum(omega * T % normal_mgf(x_j, mu, sigma) % P - mu * x_j) +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


double 
objective_sigma_exp(double sigma, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double mu = arg->mu;
    double omega = arg->omega;
    double lambda = arg->lambda;
    const arma::vec &P = arg->P;
    const arma::vec &T = arg->T;
    const arma::vec x_j = arg->x_j;
    bool verbose = arg->verbose;

    double res = sum(omega * T % normal_mgf(x_j, mu, sigma) % P - mu * x_j) +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


// [[Rcpp::export]]
double
optimise_gamma_exp(double mu, double sigma, double lambda, double omega,
	double a_0, double b_0, const arma::vec &P, const arma::vec &T, 
	const arma::vec &x_j, bool verbose) 
{
    double res = sigmoid(log(a_0 / b_0) + 1.0/2.0 -
	    (lambda * sigma * sqrt(2.0 / PI) * exp(-pow(mu/sigma, 2 )) +
	     lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) +
	     log(sqrt(2.0 / PI) * 1.0 /(sigma * lambda)) +
	     sum(omega * T % P % (normal_mgf(x_j, mu, sigma) - 1) - mu * x_j)));
    if (verbose)
	Rcpp::Rcout << "f: " << res << "\n";

    return res;
}


double
optimise_mu_exp(double sigma, double omega, double lambda, 
    const arma::vec &P, const arma::vec &T, const arma::vec x_j, bool verbose)
{
    kwargs args = {0.0, sigma, lambda, omega, P, T, x_j, verbose};
    return Brent_fmin(-1e2, 1e2, objective_mu_exp, 
	    static_cast<void*>(&args), 1e-5);
}


double
optimise_sigma_exp(double mu, double omega, double lambda, 
    const arma::vec &P, const arma::vec &T, const arma::vec x_j, bool verbose)
{
    kwargs args = {mu, 0.0, lambda, omega, P, T, x_j, verbose};
    return Brent_fmin(0, 4, objective_sigma_exp, 
	    static_cast<void*>(&args), 1e-5);
}

