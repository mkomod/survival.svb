
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

R_opt_exp_gamma <- function(mu, sig, omega, lambda, a_0, b_0, P, Y, x_j) {
    sigmoid(
	log(a_0/b_0) + 1/2 - (
	lambda * sig * sqrt(2/pi) * exp(-(mu/sig)^2) +
	lambda * mu * (1 - 2 * pnorm(-mu/sig)) +
	log(sqrt(2/pi) * 1/(sig * lambda)) +
	sum(omega*Y*P*(exp(x_j*mu + 1/2*sig^2*x_j^2) - 1) - mu*x_j) 
    ))
}

sigmoid <- function(x) 1/(1 + exp(-x))
mse <- function(x) mean(x^2)


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

mse(b - m)
mse(coef(m.1) - b)

# performance from CoxPH
m.1 <- survival::coxph(survival::Surv(Y) ~ ., data = as.data.frame(X))
