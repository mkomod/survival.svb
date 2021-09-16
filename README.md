# survival.svb

An R package for high dimensional survival analysis

## Installation

Install directly using devtools

```{r}
devtools::install_github("mkomod/survival.svb")
```

## Example

```{r}
n <- 200                        # number of sample
p <- 1000                       # number of features
s <- 10                         # number of non-zero coefficients
censoring_lvl <- 0.4            # degree of censoring


# generate some test data
set.seed(1)
b <- sample(c(runif(s, -2, 2), rep(0, p-s)))
X <- matrix(rnorm(n * p), nrow=n)
Y <- log(1 - runif(n)) / -exp(X %*% b)
delta  <- runif(n) > censoring_lvl   		# 0: censored, 1: uncensored
Y[!delta] <- Y[!delta] * runif(sum(!delta))	# rescale censored data


# fit the model
f <- survival.svb::svb.fit(Y, delta, X)


# plot the results
plot(b, xlab=expression(beta), main="Coefficient value", pch=8, ylim=c(-2,2))
points(f$m * f$g, pch=20, col=2)
legend("topleft", legend=c(expression(beta), expression(hat(beta))),
       pch=c(8, 20), col=c(1, 2))
```

