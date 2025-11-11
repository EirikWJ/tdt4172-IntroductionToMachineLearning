import numpy as np

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        m = y.shape[0]
        eps = 1e-15  # small number to prevent log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
    def compute_gradients(self, x, y, y_pred):
        m = x.shape[0]
        grad_w = (1/m) * np.dot(x.T, (y_pred - y))
        grad_b = (1/m) * np.sum(y_pred - y)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def fit(self, x, y):
        self.weights = np.zeros(x.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            lin_model = np.dot(x, self.weights) + self.bias
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

    def predict(self, x):
        y_pred = self.predict_proba(x)
        return [1 if _y > 0.5 else 0 for _y in y_pred]

    def predict_proba(self, x):
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        
    def predict_proba_roc(self, x):
        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return y_pred



