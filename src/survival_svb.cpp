#include <math.h>
#include <vector>

#include "RcppEigen.h"

#include "optimiser.hpp"
#include "partial.hpp"
#include "utils.hpp"
#include "survival.svb_types.h"


// [[Rcpp::export]]
Rcpp::List fit_partial(vec T, vec delta, mat X, double lambda, double a_0,
    double b_0, vec m, vec s, vec g, int maxiter, double tol, bool verbose)
{
    uint p = X.cols();

    std::vector<uint> delta_ord = order_delta(T, delta);

    // initialisations
    vec m_old, s_old, g_old;
    vec P = init_log_P(X, m, s, g);

    for (int iter = 1; iter <= maxiter; ++iter) {

	m_old = m; s_old = s; g_old = g;

	for (uint j = 0; j < p; ++j) {
	    vec x_j = X.col(j);

	    rm_log_P(P, x_j, m(j), s(j), g(j));

	    m(j) = opt_par_mu( s(j), lambda, P, x_j, delta_ord);
	    s(j) = opt_par_sig(m(j), lambda, P, x_j, delta_ord);
	    g(j) = opt_par_gam(m(j), s(j), lambda, a_0, b_0, P, x_j, delta_ord);

	    add_log_P(P, x_j, m(j), s(j), g(j));
	    
	    // check for overflow
	    if (!P.allFinite())
		Rcpp::stop("Overflow error. Try rescaling X or using "
			"different starting values");

	}

	// check for break
	Rcpp::checkUserInterrupt();

	// check convergence
	if ((m - m_old).cwiseAbs().sum() < tol && 
	    (s - s_old).cwiseAbs().sum() < tol && 
	    (g - g_old).cwiseAbs().sum() < tol) {
	    if (verbose)
		Rcpp::Rcout << "Converged in " << iter << " iterations\n";
	    return Rcpp::List::create(
		Rcpp::Named("mu") = m,
		Rcpp::Named("sigma") = s,
		Rcpp::Named("gamma") = g,
		Rcpp::Named("converged") = true
	    );
	}
    }
    
    if (verbose)
	Rcpp::Rcout << "Failed to converge in " << maxiter << " iterations.\n";

    return Rcpp::List::create(
	Rcpp::Named("mu") = m,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = false
    );
}

