import numpy as np


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


def l2_regularization(W, reg_strength):
    loss = reg_strength * np.sum(pow(W, 2))
    grad = 2 * reg_strength * W
    return loss, grad
    

def linear_softmax(X, W, target_index):
    predictions = np.dot(X, W)
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            for indices in batches_indices:
                batch_X, batch_y = X[indices], y[indices]
                loss, dW = linear_softmax(batch_X, self.W, batch_y)
                l2_loss, l2_dW = l2_regularization(self.W, reg)
                
                loss += l2_loss
                dW += l2_dW
                
                self.W -= learning_rate * dW
                loss_history.append(loss)
            
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        predictions = np.dot(X, self.W)
        y_pred = np.where(predictions == np.amax(predictions, axis=1, keepdims=True))[1]
        return y_pred



                
                                                          

            

                
