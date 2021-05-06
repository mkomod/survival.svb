#include "partial.hpp"
#include "math.h"


struct kwargs {
    double mu; 
    double sigma; 
    double lambda; 
    const std::vector<arma::uvec> &R;
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
    const std::vector<arma::uvec> &R = arg->R;
    const arma::uvec &F = arg->F;
    const arma::vec &P = arg->P;
    const arma::vec &x_j = arg->x_j;

    double t = 0.0;
    double s = 0.0;
    double a = max(P + log_normal_mgf(x_j, mu, sigma));
    for (auto it = R.rbegin(); it != R.rend(); ++it) {
	arma::uvec risk_set = (*it);
	// P(risk_set) = sum_j log P_j(risk_set) 
	s += sum(exp(P(risk_set) + log_normal_mgf(x_j(risk_set), mu, sigma) - a));
	t += a + log(s);
    }
    t -= mu * sum(x_j(F));

    double res = t +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2) * 0.5) +
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
    const std::vector<arma::uvec> &R = arg->R;
    const arma::uvec &F = arg->F;
    const arma::vec &P = arg->P;
    const arma::vec &x_j = arg->x_j;

    double t = 0.0;
    double s = 0.0;
    double a = max(P + log_normal_mgf(x_j, mu, sigma));
    for (auto it = R.rbegin(); it != R.rend(); ++it) {
	arma::uvec risk_set = (*it);
	s += sum(exp(P(risk_set) + log_normal_mgf(x_j(risk_set), mu, sigma) - a));
	t += a + log(s);
    }
    t -= mu * sum(x_j(F));

    double res = t +
	lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2) * 0.5) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
	log(sigma);

    return res;
}


double
opt_par_mu(double sigma, double lambda, const std::vector<arma::uvec> &R, 
    const arma::uvec &F, const arma::vec &P, const arma::vec &x_j)
{
    kwargs args = { 0.0, sigma, lambda, R, F, P, x_j };
    double res = Brent_fmin(-1e2, 1e2, f_par_mu, static_cast<void *>(&args), 1e-5);
    return res;
}


double
opt_par_sig(double mu, double lambda, const std::vector<arma::uvec> &R, 
    const arma::uvec &F, const arma::vec &P, const arma::vec &x_j)
{
    kwargs args = { mu, 0.0, lambda, R, F, P, x_j };
    double res = Brent_fmin(0.0, 10.0, f_par_sig, static_cast<void *>(&args), 1e-5);
    return res;
}


double 
opt_par_gam(double mu, double sigma, double a_0, double b_0, double lambda, 
	const std::vector<arma::uvec> &R, const arma::uvec &F, 
	const arma::vec &P, const arma::vec &x_j)
{
    double t = 0.0; double s = 0.0; double p = 0.0; double a = 0.0; double b = 0.0;
    a = std::min(max(P + log_normal_mgf(x_j, mu, sigma)) / 5.0, 175.0);
    b = std::min(max(P) / 5.0, 175.0);
    for (auto it = R.rbegin(); it != R.rend(); ++it) {
	arma::uvec risk_set = (*it);
	s += sum(exp(P(risk_set) + log_normal_mgf(x_j(risk_set), mu, sigma) - a));
	p += sum(exp(P(risk_set) - b));
	t += (a + log(s)) - (b  + log(p));
    }

    double res = sigmoid(log(a_0 / b_0) + 1.0/2.0 -
	    (lambda * sigma * sqrt(2.0 / PI) * exp(-pow(mu/sigma, 2) * 0.5) +
	     lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) +
	     log(sqrt(2.0 / PI) * 1.0 /(sigma * lambda)) + t));
    
    return res;
}


