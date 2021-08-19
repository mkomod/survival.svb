#' Compute the evidence lower bound (ELBO) for the Cox model
#'
#' @param Y failure times
#' @param delta censoring indicator
#' @param X deisgn matrix
#' @param fit fit model
#' @param nrep Monte Carlo samples
#'
#' @export
elbo <- function(Y, delta, X, fit, nrep=1e4)
{
    p <- ncol(X)
    m <- fit$m
    s <- fit$s
    g <- fit$g
    lambda <- fit$lambda
    a0 <- fit$a0
    b0 <- fit$b0

    res.likelihood <- replicate(nrep, {
	b. <- (runif(p) < g) * rnorm(p, m, s)
	log_likelihood(Y, delta, X, b.)
    }) 
    res.kl <- sum(
	g * (lambda * s * sqrt(2/pi)*exp(-(m^2)/(2*s^2)) + 
	     lambda * m * (1 - 2*pnorm(-m / s)) +
	     log(sqrt(2) / (sqrt(pi) * s * lambda)) -
	     0.5 -
	     log(a0 / b0)) +
	g * log(g) + (1-g)*log(1 - g + 1e-18) - log(b0 / (a0 + b0))
    )

    m.res.likelihood <- mean(res.likelihood)
    s.res.likelihood <- sd(res.likelihood)

    return(list(mean=m.res.likelihood - res.kl, 
		sd=s.res.likelihood,
		expected.likelihood=m.res.likelihood, 
		kl.Q_Pi=res.kl))
}
