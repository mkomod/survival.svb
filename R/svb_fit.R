#' Fit sparse variational Bayesian proportional hazards model.
#'
#' @param Y Failure times.
#' @param delta Censoring indicator, 0: censored, 1: uncensored.
#' @param X Design matrix.
#' @param lambda Penalisation parameter, default: \code{lambda=0.5}.
#' @param a0 Beta distribution parameter, default: \code{a0=1}.
#' @param b0 Beta distribution parameter, default: \code{b0=ncol(X)}.
#' @param mu.init Initial value for the means of the Gaussian component of the variational family (\eqn{\mu}), default taken from LASSO fit.
#' @param s.init Initial value for the standard deviations of the Gaussian component of the variational family (\eqn{s}), default: \code{rep(0.05, ncol(X))}.
#' @param g.init Initial value for the inclusion probability (\eqn{\gamma}), default: \code{rep(0.5, ncol(X))}.
#' @param maxiter Maximum number of iterations, default: \code{1000}.
#' @param tol Convergence tolerance, default: \code{0.001}.
#' @param alpha The elastic-net mixing parameter used for initialising \code{mu.init}. When \code{alpha=1} the lasso penalty is used and \code{alpha=0} the ridge penalty, values between 0 and 1 give a mixture of the two penalties, default: \code{1}.
#' @param center Center X prior to fitting, increases numerical stability, default: \code{TRUE}
#' @param verbose Print additional information: default: \code{TRUE}.
#'
#' @examples
#' \dontrun{
#' n <- 200                        # number of sample
#' p <- 1000                       # number of features
#' s <- 10                         # number of non-zero coefficients
#' censoring_lvl <- 0.4            # degree of censoring
#' 
#' 
#' # generate some test data
#' set.seed(1)
#' b <- sample(c(runif(s, -2, 2), rep(0, p-s)))
#' X <- matrix(rnorm(n * p), nrow=n)
#' Y <- log(1 - runif(n)) / -exp(X %*% b)
#' delta  <- runif(n) > censoring_lvl   		# 0: censored, 1: uncensored
#' Y[!delta] <- Y[!delta] * runif(sum(!delta))	# rescale censored data
#' 
#' 
#' # fit the model
#' f <- survival.svb::svb.fit(Y, delta, X)
#' 
#' 
#' # plot the results
#' plot(b, xlab=expression(beta), main="Coefficient value", pch=8, ylim=c(-2,2))
#' points(f$m * f$g, pch=20, col=2)
#' legend("topleft", legend=c(expression(beta), expression(hat(beta))),
#'        pch=c(8, 20), col=c(1, 2))
#' plot(f$g, main="Inclusion Probabilities", ylab=expression(gamma))
#' }
#' @export
svb.fit <- function(Y, delta, X, lambda=0.5, a0=1, b0=ncol(X),
    mu.init=NULL, s.init=rep(0.05, ncol(X)), g.init=rep(0.5, ncol(X)), 
    maxiter=1e3, tol=1e-3, alpha=1, center=TRUE, verbose=TRUE)
{
    if (!is.matrix(X)) stop("'X' must be a matrix")
    if (!(lambda > 0)) stop("'lambda' must be greater than 0")
    
    p <- ncol(X)
    if (is.null(mu.init)) {
	y <- survival::Surv(as.matrix(Y), as.matrix(as.numeric(delta)))
	g <- glmnet::glmnet(X, y, family="cox", nlambda=10, alpha=alpha,
	    standardize=FALSE)

	# use the fit for the smallest value of glmnet's lambda seq
	mu.init <- g$beta[ , ncol(g$beta)]
    }
    
    # re-order Y, delta and X by failure time
    # log-likelihood computed with sorted data
    oY <- order(Y)
    Y <- Y[oY]
    delta <- delta[oY]
    X <- X[oY, ]

    if (center) {
	X <- scale(X, center=T, scale=F)
    }

    # fit the model
    res <- fit_partial(Y, delta, X, lambda, a0, b0,
	mu.init, s.init, g.init, maxiter, tol, verbose)

    return(c(res, beta_hat=res$m * res$g, lambda=lambda, a0=a0, b0=b0))
}
