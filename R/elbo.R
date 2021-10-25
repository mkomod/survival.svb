#' Compute the evidence lower bound (ELBO)
#'
#' @param Y Failure times.
#' @param delta Censoring indicator, 0: censored, 1: uncensored.
#' @param X Design matrix.
#' @param fit Fit model.
#' @param nrep Number of Monte Carlo samples.
#' @param center Should the design matrix be centered.
#'
#' @return Returns a list containing: \cr
#' \item{mean}{The mean of the ELBO.}
#' \item{sd}{The standard-deviation of the ELBO.}
#' \item{expected.likelihood}{The expectation of the likelihood
#' under the variational posterior.}
#' \item{kl}{The KL between the variational posterior and prior.}
#'
#' @section Details:
#' The evidence lower bound (ELBO) is a popular goodness of fit measure
#' used in variational inference. Under the variational posterior the
#' ELBO is given as
#' \deqn{\text{ELBO} = E_{\tilde{\Pi}}[\log L_p(\beta; Y, X, \delta)] - \text{KL}(\tilde{\Pi} \| \Pi)}
#' where \eqn{\tilde{\Pi}} is the variational posterior, \eqn{\Pi} is the prior,
#' \eqn{L_p(\beta; Y, X, delta)} is Cox's partial likelihood. Intuitively,
#' within the ELBO we incur a trade-off between how well we fit to the data
#' (through the expectation of the log-partial-likelihood) and how close we
#' are to our prior (in terms of KL divergence). Ideally we want the ELBO to be 
#' as small as possible.
#'
#' @export
elbo <- function(Y, delta, X, fit, nrep=1e4, center=TRUE)
{
    p <- ncol(X)
    m <- fit$m
    s <- fit$s
    g <- fit$g
    lambda <- fit$lambda
    a0 <- fit$a0
    b0 <- fit$b0

    if (center)
	X <- scale(X, center=TRUE, scale=FALSE)

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
		kl=res.kl))
}
