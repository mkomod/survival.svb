#include <vector>

#include "RcppArmadillo.h"


// [[Rcpp::export]]
Rcpp::List
mc_eb(arma::vec T, arma::vec delta, arma::mat X, arma::vec m, 
	arma::vec s, arma::vec g, int mc_samples, bool verbose)
{
    int p = X.n_cols;

    // indices of failure times
    arma::uvec F = find(delta);
    
    // construct list of indices between two failure times
    arma::vec FT = sort(T(F));
    std::vector<arma::uvec> R;
    for (int i = 0; i < FT.size() - 1; ++i)
	R.push_back(find(T < FT(i+1) && T >= FT(i)));
    R.push_back(find(T >= FT(FT.size() - 1)));
    
    arma::vec mg = g % m;
    double ev = 0.0;
    
    for (int i = 0; i < mc_samples; ++i) {
	arma::vec b = arma::vec(p, arma::fill::zeros);
	for (int j = 0; j < p; ++i) {
	    if (R::runif(0, 1) <= g(j))
		b(j) = R::rnorm(mg(j), s(j));
	}
	arma::vec Xb = X * b;

	double mr = 0.0;
	double pv = 0.0;
	double mv = 0.0;
	for (auto it = R.rbegin(); it != R.rend(); ++it) {
	    arma::uvec risk_set = (*it);
	    mv = max(Xb(risk_set));
	    pv = pv > mv ? pv : mv;
	    mr += pv;
	}

	ev += exp(sum(Xb(F)) - mr);
    }

    return Rcpp::List::create(
	Rcpp::Named("ev") = (ev / mc_samples)
    );
}


