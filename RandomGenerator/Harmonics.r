library(rstan)
set.seed(1)

warmup = 100
iter = 200

l = 1
m = -1

stan_data <- list(l = l, m = m)

MCMC_Sampler <- stan_model(file = "C:\\Users\\amval\\.vscode-R\\Harmonics.stan")

for (i in 1:20) {

    file <- "C:\\Users\\amval\\PyWork\\Harmonics\\True_Samples_l1m1\\Harmonic_Samples"
    samples = suppressMessages(sampling(MCMC_Sampler, data = stan_data, warmup = warmup,
                        iter = iter, chains = 1000, seed = i))

    theta <- extract(samples, pars = 'theta')[[1]]
    theta <- sample(theta, 5000, replace = FALSE)
    phi <- runif(5000, min = 0, max = 3.14159)
    #hist(theta)
    Data <- data.frame(Theta = theta, Phi = phi)
    file <- paste0(file, sep = "_", i)
    file <- paste0(file, ".csv")
    write.csv(Data, file, row.names = FALSE)

}


for (i in 1:20) {

    file <- "C:\\Users\\amval\\PyWork\\Harmonics\\True_HalfUniform\\Uniform_Samples"

    theta <- runif(5000, min = 0, max = pi)
    theta <- sample(theta, 5000, replace = FALSE)
    phi <- runif(5000, min = 0, max = pi)
    #hist(theta)
    Data <- data.frame(Theta = theta, Phi = phi)
    file <- paste0(file, sep = "_", i)
    file <- paste0(file, ".csv")
    write.csv(Data, file, row.names = FALSE)

}
for (i in 1:20) {

    file <- "C:\\Users\\amval\\PyWork\\Harmonics\\False_HalfUniform\\Uniform_Samples"

    theta <- runif(5000, min = pi, max = 2*pi)
    theta <- sample(theta, 5000, replace = FALSE)
    phi <- runif(5000, min = 0, max = pi)
    #hist(theta)
    Data <- data.frame(Theta = theta, Phi = phi)
    file <- paste0(file, sep = "_", i)
    file <- paste0(file, ".csv")
    write.csv(Data, file, row.names = FALSE)

}


library(truncnorm)
for (i in 1:20) {

    file <- "C:\\Users\\amval\\PyWork\\Harmonics\\Normal_Samples\\Normal_Samples"

    theta <- rtruncnorm(5000, a = 0, b = 2*pi, mean = pi, sd = 1)
    theta <- sample(theta, 5000, replace = FALSE)
    phi <- rtruncnorm(5000, a = 0, b = pi, mean = pi/2, sd = 1)
    #hist(theta)
    Data <- data.frame(Theta = theta, Phi = phi)
    file <- paste0(file, sep = "_", i)
    file <- paste0(file, ".csv")
    write.csv(Data, file, row.names = FALSE)

}