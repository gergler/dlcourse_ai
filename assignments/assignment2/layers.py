import numpy as np


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(pow(W, 2))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax(predictions):
    copy_predictions = np.copy(predictions)
    if predictions.ndim == 1:
        copy_predictions -= np.max(predictions) # numerical stability
        probs = np.exp(copy_predictions)/np.sum(np.exp(copy_predictions))
    else: 
        copy_predictions -= np.amax(predictions, axis=1, keepdims=True) # numerical stability
        probs = np.exp(copy_predictions)/np.sum(np.exp(copy_predictions), axis=1, keepdims=True)
        
    return probs


def cross_entropy_loss(probs, target_index):
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        batch_loss = -np.log(probs[np.arange(batch_size), target_index.flatten()]) # return a copy of the array collapsed into one dimension
        loss = np.sum(batch_loss)/batch_size
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    
    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else: 
        batch_size = predictions.shape[0]
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1 
        dprediction /= batch_size
        
    return loss, dprediction


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
        return X * (X > 0)

    def backward(self, d_out):
        d_result = d_out * (self.X > 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.W.value) + self.B.value 

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis = 0, keepdims = True)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
