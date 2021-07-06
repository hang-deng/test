import numpy as np
import scipy.optimize


class OBLR():
    '''
    This class implements online Bayesian Logistic Regression model.
    '''

    def __init__(self, mu, q, alpha=1, C=1):
        """
        :param mu: Gaussian prior mean, that is, mean = mu_i. Same length with q.
        :param q: Gaussian prior variance inverse, that is, variance = 1/q_i. Same length with mu
        :param alpha: when sampling parameters based on posterior, we shrink variance by alpha to control exploration
        :param C: l_2 penalty coefficient in regularized logistic regression
        :param dim: dimension of the entire feature vector
        """
        self.mu = mu
        self.q = q
        self.alpha = alpha
        self.C = C
        self.dim = len(self.mu)
        self._config = {'mu': self.mu, 'q': self.q, "alpha": self.alpha, "C": self.C, "dim": self.dim}

    def fit(self, X, y):
        """
        Update the model
        :param X: a matrix of shape n * self.dim, where n is the number of observations
        :param y: a vector of size n
        :return: None
        """
        self.mu = scipy.optimize.minimize(lambda w: self._objective(w, X, y), x0=self.mu, method="BFGS").x
        p = 1 / (1 + np.exp(- X.dot(self.mu)))
        self.q = self.q + (X ** 2).transpose().dot(p * (1 - p))

    def _objective(self, w, X, y):
        return self.q.dot((self.mu - w) ** 2) / 2 + np.log(1 + np.exp(- X.dot(w) * y)).sum()

    def predict(self):
        pass

    def predict_proba(self):
        pass

    def get_params(self, display=False):
        """
        :return: current values of the parameters
        """
        if display:
            print(f"The current parameters of the model are: \n "
                  f"Mean of posteriors: {self.mu} \n "
                  f"Variance inverse of posteriors: {self.q} \n"
                  f"Alpha: {self.alpha}, C: {self.C}")
        return self.mu, self.q, self.alpha, self.C

    def _get_config(self):
        """
        :return: original configuration of parameters
        """
        return self._config


"""
Test if the algorithm yields accurate result
"""

MU = np.zeros(100)
Q = np.ones(100)

np.random.seed(1)

W = np.random.normal(0.2, 1, 100)

model = OBLR(mu=MU, q=Q)
print(f"Initial: {((model.mu - W) ** 2).sum()}")

for i in range(1000):
    # create a new user profile
    user = np.random.normal(0, 1, 50).reshape((1, 50))

    # create 50 new items
    item = np.random.normal(0, 1, 50 * 50).reshape((50, 50))

    # combine user profile and item features, 50 * 100
    feature = np.concatenate((np.broadcast_to(user, (50, 50)), item), axis = 1)

    # calculate true probabilities
    p = 1/(1 + np.exp(- feature.dot(W)))

    # generate label based on p
    label = np.random.binomial(1, p)

    # fit data into model
    model.fit(feature, label)

    # print the distance between current parameters to true parameters
    # mean, variance, alpha, C = model.get_params()
    if i % 50 == 0:
        print(f"Step {i}: {((model.mu - W) ** 2).sum()}")
