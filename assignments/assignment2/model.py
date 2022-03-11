import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.fully_conect_layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()
        self.fully_conect_layer_2 = FullyConnectedLayer(hidden_layer_size, n_output) 
        self.reg = reg

    def compute_loss_and_gradients(self, X, y):
        for param in self.params().values():
            param.grad = 0
        
        forward_layer_1 = self.fully_conect_layer_1.forward(X)
        forward_relu_layer = self.relu_layer.forward(forward_layer_1)
        forward_layer_2 = self.fully_conect_layer_2.forward(forward_relu_layer)
        
        loss, grad = softmax_with_cross_entropy(forward_layer_2, y)
        
        backward_layer_2 = self.fully_conect_layer_2.backward(grad)
        backward_relu_layer = self.relu_layer.backward(backward_layer_2)
        backward_layer_1 = self.fully_conect_layer_1.backward(backward_relu_layer)

        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg) 
            loss += reg_loss
            param.grad += reg_grad

        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0], np.int)
        
        forward_layer_1 = self.fully_conect_layer_1.forward(X)
        forward_relu_layer = self.relu_layer.forward(forward_layer_1)
        forward_layer_2 = self.fully_conect_layer_2.forward(forward_relu_layer)
        
        pred = np.argmax(forward_layer_2, axis=1)

        return pred

    def params(self):
        result = {'W1': self.fully_conect_layer_1.params()['W'], 
                  'B1': self.fully_conect_layer_1.params()['B'], 
                  'W2': self.fully_conect_layer_2.params()['W'], 
                  'B2': self.fully_conect_layer_2.params()['B']}

        return result
