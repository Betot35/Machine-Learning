import numpy as np

class Perceptron(object):
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iterations):
            errors = 0
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                update = self.learning_rate * (expected_value - predicted_value)
                self.coef_[1:] += update * xi
                self.coef_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    def activation_function(self, X):
        weighted_sum = self.net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)

    def predict(self, X):
        return self.activation_function(X)

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if target != output:
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_

if __name__ == "__main__":
    X = np.array([[2, 3], [1, 2], [3, 4], [5, 5]])
    y = np.array([0, 0, 1, 1])  
    perceptron = Perceptron(n_iterations=100, learning_rate=0.1)
    perceptron.fit(X, y)

    new_data = np.array([[4, 3]])
    prediction = perceptron.predict(new_data)

    if prediction == 0:
        print("The new data point belongs to class 0.")
    else:
        print("The new data point belongs to class 1.")