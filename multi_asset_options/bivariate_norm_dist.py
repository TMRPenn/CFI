from scipy.stats import multivariate_normal

def bivariate_normal_cdf(a, b, p):
    # Define the mean and covariance matrix for the bivariate normal distribution
    mean = [0, 0]
    cov = [[1, p], [p, 1]]

    # Create a multivariate normal distribution object
    rv = multivariate_normal(mean, cov)

    # Calculate the cumulative distribution function for the given limits a and b
    prob = rv.cdf([a, b])
    return prob


if __name__ == "__main__":
    # Example usage:
    a = 1.0
    b = 1.0
    correlation = 0.2
    result = bivariate_normal_cdf(a, b, correlation)
    print(f"The cumulative bivariate normal distribution M({a}, {b}; {correlation}) is: {result}")

    prob = multivariate_normal([0, 0], [[1, correlation], [correlation, 1]]).cdf([a, b])
    print(prob)