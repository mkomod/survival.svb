#include "RcppArmadillo.h"
#include "utils.hpp"

class ExponentialMean {
    private:
	arma::colvec P;
	arma::colvec x_j;
	arma::colvec T;
	double omega;
	double sigma;
	double lambda;


    public:
	double Evaluate(const double mu) {
	    return sum(omega * T % normal_mgf(x_j, mu, sigma) % P - mu * x_j) +
		lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
		lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0));
	}

	void Gradient(const double mu, double &g) {
	    g = sum(omega * T % P % normal_mgf(x_j , mu, sigma) % x_j - x_j) -
		2.0 * lambda * mu/sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
		lambda * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) +
		lambda * 2.0 / sigma * R::dnorm(- mu / sigma, 0, 1, 0);
	}
};


class ExponentialVariance {
    private:
	arma::colvec P;
	arma::colvec x_j;
	arma::colvec T;
	double omega;
	double mu;
	double lambda;


    public:
	double Evaluate(const double sigma) {
	    return sum(omega * T % normal_mgf(x_j, mu, sigma) % P - mu * x_j) +
		lambda * sigma * sqrt(2.0/PI) * exp(-pow(mu/sigma, 2)) +
		lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) -
		log(sigma);
	}

	void Gradient(const double sigma, double &g) {
	    g = sum(omega * T % P % normal_mgf(x_j , mu, sigma) % x_j * sigma) -
		(1.0 + 2.0 * mu / pow(sigma, 2)) * lambda * sqrt(2.0/PI) *  
		exp(-pow(mu/sigma, 2)) -
		2.0 * lambda * mu/pow(sigma, 2)* R::dnorm(- mu / sigma, 0, 1, 0) +
		1 / sigma;
	}
};


double
ExponentialWeights(double mu, double sigma, double lambda, double omega,
	double a_0, double b_0, arma::vec P, arma::vec T, arma::vec x_j) 
{
    return sigmoid(log(a_0 / b_0) + 1.0/2.0 -
	    (lambda * sigma * sqrt(2.0 / PI) * exp(-pow(mu/ sigma, 2 )) +
	     lambda * mu * (1.0 - 2.0 * R::pnorm(- mu / sigma, 0, 1, 1, 0)) +
	     log(sqrt(2.0 / PI) * 1.0 /(sigma * lambda)) +
	     sum(omega * T % P % (normal_mgf(x_j, mu, sigma) - 1) - mu * x_j)));
}
