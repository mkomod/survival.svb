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
    m <- fit$m; s <- fit$s; g <- fit$g
    lambda <- fit$lambda; a0 <- fit$a0; b0 <- fit$b0
    w <- a0 / (a0 + b0)

    res <- sum(
	(1-g)*log(1 - g + 1e-18) - (1-g)*log(1 - w) +
	g*log(g/w) - 0.5*g*log(2*pi*s^2) - g/2 -
	g*log(lambda/2) + lambda*g*s*sqrt(2/pi)*exp(-(m^2)/(2*s^2)) + 
	lambda*g*m*(1-2*pnorm(-m/s))
    ) -
    replicate(nrep, {
	b. <- (runif(p) < g) * rnorm(p, m, s)
	log_likelihood(b., X, Y, delta)
    })

    return(list(mean=mean(res), sd=sd(res)))
}
