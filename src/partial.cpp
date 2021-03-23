#include "partial.hpp"


struct kwargs {
    double mu; 
    double sigma; 
    double lambda; 
    const arma::vec &T;
    const arma::uvec &F; 
    const arma::vec &P;
    const arma::vec &x_j;
};


static double 
f_par_mu(double mu, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double sigma = arg->sigma;
    double lambda = arg->lambda;
    const arma::vec &T = arg->T;
    const arma::uvec &F = arg->F;
    const arma::vec &P = arg->P;
    const arma::vec &x_j = arg->x_j;

    double t = 0.0;
    for (auto i : F) {
	arma::uvec risk_set = find(T >= T(i));
	double m = log(sum(
	    P(risk_set) % 
	    normal_mgf(x_j(risk_set), mu, sigma)
	));
	t += m - mu * x_j(i);
    }

    double res = t +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    return res;
}


static double
f_par_sig(double sigma, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double mu = arg->mu;
    double lambda = arg->lambda;
    const arma::vec &T = arg->T;
    const arma::uvec &F = arg->F;
    const arma::vec &P = arg->P;
    const arma::vec &x_j = arg->x_j;

    double t = 0.0;
    for (auto i : F) {
	arma::uvec risk_set = find(T >= T(i));
	double m = log(sum(
	    P(risk_set) % 
	    normal_mgf(x_j(risk_set), mu, sigma)
	));
	t += m - mu * x_j(i);
    }

    double res = t +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    return res;
}


double
opt_par_mu(double sigma, double lambda, const arma::vec &T, 
    const arma::uvec &F, const arma::vec &P, const arma::vec &x_j)
{
    kwargs args = {0.0, sigma, lambda, T, F, P, x_j };
    return Brent_fmin(-1e2, 1e2, f_par_mu, static_cast<void *>(&args), 1e-5);
}


double
opt_par_sig(double mu, double lambda, const arma::vec &T, 
    const arma::uvec &F, const arma::vec &P, const arma::vec &x_j)
{
    kwargs args = {mu, 0.0, lambda, T, F, P, x_j };
    return Brent_fmin(0.0, 10.0, f_par_mu, static_cast<void *>(&args), 1e-5);
}


double 
opt_par_gam(double mu, double sigma, double a_0, double b_0, double lambda, 
	const arma::vec &T, const arma::uvec &F, const arma::vec &P, 
	const arma::vec &x_j)
{
    double t = 0.0; 
    for (auto i : F) {
	arma::uvec risk_set = find(T >= T(i));
	double m = sum(
	    P(risk_set) % normal_mgf(x_j(risk_set), mu, sigma)
	);
	t += log(m) - log(sum(P(risk_set))) - mu * x_j(i);
    }

    double res = sigmoid(log(a_0 / b_0) + 1.0/2.0 -
	    (lambda * sigma * sqrt(2.0 / PI) * exp(-pow(mu/sigma, 2 )) +
	     lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) +
	     log(sqrt(2.0 / PI) * 1.0 /(sigma * lambda)) + t));

    return res;
}


