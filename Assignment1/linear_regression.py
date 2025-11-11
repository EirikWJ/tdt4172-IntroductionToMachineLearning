import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.01, epochs=1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit  
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w, self.b = 0, 0
        self.losses = []

    def compute_gradients(self, X, y, y_pred):
        m = X.shape[0]
        grad_b,grad_w = 0,0
        for i in range(m):
            grad_b += 2/m * (y_pred[i] - y[i])
            grad_w += 2/m * (X[i] * (y_pred[i] - y[i]))
        return grad_b, grad_w

    
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        
        X = np.array(X)
        y = np.array(y)
        
        for _ in range(self.epochs):
            y_pred = self.w * X + self.b
            grad_b, grad_w = self.compute_gradients(X, y, y_pred)
            self.b -= self.learning_rate * grad_b
            self.w -= self.learning_rate * grad_w
            
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
         # TODO: Implement
        return self.w * X + self.b



