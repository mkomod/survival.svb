#include "partial.h"


struct kwargs {
    double mu; 
    double sigma; 
    double lambda; 
    const vec &P;
    const vec &x_j;
    const std::vector<unsigned int> &delta_ord;
};


// [[Rcpp::export]]
double pm(double mu, double sigma, const vec &P, const vec &x_j,
	const std::vector<unsigned int> &delta_ord) 
{
    double s = 0.0;
    double t = 0.0;
    double a = 0.0;

    a = (P + log_normal_mgf(x_j, mu, sigma)).maxCoeff();
    unsigned int icf = 0;		// index of current failure
    unsigned int ilf = x_j.rows();	// index of last failure

    for (int i = delta_ord.size() - 1; i >= 0; --i) {
	icf = delta_ord.at(i);

	for (unsigned int j = icf; j <= ilf - 1; ++j)
	    s += exp(P(j) + log_normal_mgf(x_j(j), mu, sigma) - a);
	
	t += a + log(s) - mu * x_j(icf);

	ilf = icf;
    }

    return t;
}


static double f_par_mu(double mu, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double sigma = arg->sigma;
    double lambda = arg->lambda;
    const vec &P = arg->P;
    const vec &x_j = arg->x_j;
    const std::vector<unsigned int> &delta_ord = arg->delta_ord;

    double t = pm(mu, sigma, P, x_j, delta_ord);

    double res = t +
	lambda * sigma * sqrt(2.0/M_PI) * exp(-pow(mu/sigma, 2.0) * 0.5) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0));

    return res;
}


static double f_par_sig(double sigma, void* args)
{
    kwargs *arg = static_cast<kwargs *>(args);
    double mu = arg->mu;
    double lambda = arg->lambda;
    const vec &P = arg->P;
    const vec &x_j = arg->x_j;
    const std::vector<unsigned int> &delta_ord = arg->delta_ord;
    
    double t = pm(mu, sigma, P, x_j, delta_ord);

    double res = t +
	lambda * sigma * sqrt(2.0/M_PI) * exp(-pow(mu/sigma, 2.0) * 0.5) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) -
	log(sigma);

    return res;
}


// [[Rcpp::export]]
double opt_par_mu(double mu, double sigma, double lambda, const vec &P, const vec &x_j,
	const std::vector<unsigned int> &delta_ord)
{
    kwargs args = { mu, sigma, lambda, P, x_j, delta_ord };
    double res = Brent_fmin(mu-2.0*abs(mu)-0.5, mu+2.0*abs(mu)+0.5, 
	    f_par_mu, static_cast<void *>(&args), 1e-5);
    return res;
}


// [[Rcpp::export]]
double opt_par_sig(double sigma, double mu, double lambda, const vec &P, const vec &x_j,
	const std::vector<unsigned int> &delta_ord)
{
    kwargs args = { mu, sigma, lambda, P, x_j, delta_ord };
    double res = Brent_fmin(1e-4, 2.0*sigma, f_par_sig, static_cast<void *>(&args), 1e-5);
    return res;
}


// [[Rcpp::export]]
double opt_par_gam(double mu, double sigma, double lambda, double a_0, double b_0,
    const vec &P, const vec &x_j, const std::vector<unsigned int> &delta_ord)
{
    double t = 0.0; 
    double s = 0.0; 
    double p = 0.0; 
    double a = 0.0; 
    double b = 0.0;

    a = std::min((P + log_normal_mgf(x_j, mu, sigma)).maxCoeff() / 5.0, 175.0);
    b = std::min(P.maxCoeff() / 5.0, 175.0);
    unsigned int icf = 0;
    unsigned int ilf = x_j.rows();

    for (int i = delta_ord.size() - 1; i >= 0; --i) {
	icf = delta_ord.at(i);

	for (unsigned int j = icf; j <= ilf - 1; ++j) {
	    s += exp(P(j) + log_normal_mgf(x_j(j), mu, sigma) - a);
	    p += exp(P(j) - b);
	}
	
	t += (a + log(s)) - (b + log(p)) - mu * x_j(icf);

	ilf = icf;
    }

    double res = sigmoid(log(a_0 / b_0) + 0.5 - (
	lambda * sigma * sqrt(2.0 / M_PI) * exp(-pow(mu/sigma, 2.0) * 0.5) +
	lambda * mu * (1.0 - 2.0 * R::pnorm(- mu/sigma, 0, 1, 1, 0)) +
	log(sqrt(2.0 / M_PI) * 1.0 /(sigma * lambda)) + t)
    );
    
    return res;
}

