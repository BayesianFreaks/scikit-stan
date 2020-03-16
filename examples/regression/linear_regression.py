def linear_regression_stan_backend():
    """
    Logistic Regression example code using Stan backend.
    """
    from skstan.model.lgm import LinearRegression

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split

    Boston = load_boston()
    X = Boston.data
    y = Boston.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    lr = LinearRegression(chains=3,
                          warmup=1000,
                          n_itr=5000,
                          n_jobs=1,
                          algorithm='NUTS',
                          verbose=False,
                          shrinkage=20,
                          sigma_upper=20)

    # train
    lr.fit(X_train, y_train)

    lr.predict(X_test)
