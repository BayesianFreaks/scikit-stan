data{
    int n;
    int f;
    matrix[n,f] x;
    vector[n] y;
    real shrinkage;
    real sigma_upper;
}
parameters{
    vector[f] alpha;
    real beta;
    real<lower=0> sigma;
}
transformed parameters{
    vector[n] yp;
    yp <- x*alpha + beta;
}
model{
    alpha ~ normal(0, shrinkage);
    beta ~ normal(0, shrinkage);
    sigma ~ uniform(0, sigma_upper);
    y ~ normal(yp, sigma);
}
