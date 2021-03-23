
# test initiailse_P
all(init_P(X, m, s, g) - 
    apply(X, 1, function(x) prod(g * exp(x * m + 1/2 * x^2 * s^2)  + (1- g)))
    < 1e-4)

# test sub_from_P, add_to_P
P <- init_P(X, m, s, g)
all(P - 
    add_P(rm_P(P, X[, 1], m[1], s[1], g[1]), X[ ,1], m[1], s[1], g[1])
    < 1e-5)


# test exp_obj
mu1 <- m[1]
sig1 <- s[1]
x_j <- X[ , 1]
lambda <- 2
obj_exp(mu1, sig1, omega, lambda, P, Y, x_j, T)
R_exp_obj(mu1, sig1, omega, lambda, P, Y, x_j)


# test opt_exp_gamma
opt_exp_gamma(mu1, sig1, lambda, omega, 1, 1, P, Y, x, 1)
R_opt_exp_gamma(mu1, sig1, omega, lambda, 1, 1, P, Y, x)


R_exp_obj <- function(mu, sig, omega, lambda, P, Y, x_j) {
    sum(omega*Y*exp(x_j*mu + 1/2*sig^2*x_j^2)*P - mu*x_j) + 
    lambda*sig*sqrt(2/pi)*exp(-(mu/sig)^2) + 
    lambda*mu*(1-2*pnorm(-mu/sig)) -
    log(sig)
}

sigmoid <- function(x) 1/(1 + exp(-x))
R_opt_exp_gamma <- function(mu, sig, omega, lambda, a_0, b_0, P, Y, x_j) {
    sigmoid(
	log(a_0/b_0) + 1/2 - (
	lambda * sig * sqrt(2/pi) * exp(-(mu/sig)^2) +
	lambda * mu * (1 - 2 * pnorm(-mu/sig)) +
	log(sqrt(2/pi) * 1/(sig * lambda)) +
	sum(omega*Y*P*(exp(x_j*mu + 1/2*sig^2*x_j^2) - 1) - mu*x_j) 
    ))
}


P <- init_P(X, m, s, g)
for (iter in 1:100) {
    for (j in 1:ncol(X)) {
	mu <- m[j]
	sig <- s[j]
	gam <- g[j]
	x_j <- X[ , j]
	
	P <- rm_P(P, x_j, mu, sig, gam)
	m[j] <- optimize(function(mu) 
		 obj_exp(mu, sig, omega, lambda, P, Y, x_j, T),
		 c(-1e2, 1e2), maximum=F)$min
	mu <- m[j]
	P <- add_P(P, x_j, mu, sig, gam)
	P <- rm_P(P, x_j, mu, sig, gam)
	s[j] <- optimize(function(sig) 
		    obj_exp(mu, sig, omega, lambda, P, Y, x_j, T),
		    c(0, 4), maximum=F)$min
	sig <- s[j]
	P <- add_P(P, x_j, mu, sig, gam)
	P <- rm_P(P, x_j, mu, sig, gam)
	g[j] <- opt_exp_gamma(mu, sig, lambda, omega, 1, 1, P, Y, x_j, 1)
	gam <- g[j]
	P <- add_P(P, x_j, mu, sig, gam)
    }
}


Rcpp::sourceCpp("../src/survival_svb.cpp", verbose=T, rebuild=T)

n <- 250
censoring_lvl <- 0.4
set.seed(1)
b <- c(1, 1, rep(0, 50))
p <- length(b)
X <- matrix(rnorm(n * p), nrow=n)
y <- runif(nrow(X))
omega <- 1
Y <- log(1 - y) / - (exp(X %*% b) * omega)
delta  <- runif(n) > censoring_lvl   # 0: censored, 1: uncensored
Y[!delta] <- Y[!delta] * runif(sum(!delta))

# params
m <- matrix(rnorm(p), ncol=1)
s <- matrix(abs(rnorm(p)), ncol=1)
g <- matrix(runif(p), ncol=1)

res <- fit_partial(Y, delta, X, lambda, 1000, T)

lambda <- 0.5
P <- init_P(X, m, s, g)
for (iter in 1:10) {
    for (j in 1:p) {
	x_j <- X[ , j]
	# sum(P)
	P <- rm_P(P, x_j, m[j], s[j], g[j])
	# sum(P)
	m[j] <- optimize(function(mu) 
		objective_mu_sig(mu, s[j], lambda, Y, res$T, P, x_j),
		c(-100, 100))$min
	s[j] <- optimize(function(sig) 
		objective_mu_sig(m[j], sig, lambda, Y, res$T, P, x_j),
		c(0, 10))$min
	g[j] <- opt_gamma(m[j], s[j], 0.001, 0.001, lambda, Y, res$T, P,
			  x_j)
	# g[j] <- optimise(function(gam)
	# 	objective_gamma(gam, m[j], s[j], 0.001, 0.001, lambda,
	# 	Y, res$T, P, x_j),
	# 	c(0, 1))$min
	P <- add_P(P, x_j, m[j], s[j], g[j])
	# sum(P)
    }
}

opt_gamma(m[j], s[j], 0.001, 0.001, lambda, Y, res$T, P, x_j)


