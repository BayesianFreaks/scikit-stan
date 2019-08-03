def linear_regression_stan_backend():
    """
    Logistic Regression example code using Stan backend.
    """
    from skstan.model.lgm import LinearRegression

    lr = LinearRegression(
        chains=3,
        warmup=1000,
        n_itr=5000,
        n_jobs=1,
        algorithm='NUTS',
        verbose=False,
        shrinkage=20
    )
    lr.fit()
