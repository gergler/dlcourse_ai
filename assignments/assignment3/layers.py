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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
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
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, 
                                       filter_size,
                                       in_channels, 
                                       out_channels))

        self.B = Param(np.zeros(out_channels))
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1
        
        self.X = np.pad(X, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = 0)
        W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        
        out_shape = (batch_size, out_height, out_width, self.out_channels)
        out_data = np.zeros(out_shape)
        
        for y in range(out_height):
            for x in range(out_width):
                h = y + self.filter_size
                w = x + self.filter_size
                in_data = self.X[:, y:h, x:w, :].reshape(batch_size, -1)
                out_data[:, y, x, :] = np.dot(in_data, W)
                
        return out_data + self.B.value


    def backward(self, d_out):       
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        pad_width, filter_size = self.padding, self.filter_size
        
        if self.padding:
            npad = ((0, 0), (pad_width, pad_width), (pad_width, pad_width), (0, 0))
            self.X = np.pad(self.X, npad, 'constant', constant_values=0)
        
        dW = np.zeros(self.W.value.shape)
        dX = np.zeros(self.X.shape)
        W_reshaped = self.W.value.reshape(-1, self.out_channels)
        
        for y in range(out_height):
            for x in range(out_width):
                h = y + self.filter_size
                w = x + self.filter_size
                
                dW += self.X[:, y:h, x:w, :].reshape(batch_size, -1).T.dot(d_out[:, y, x, :])\
                .reshape(self.filter_size, self.filter_size, channels, out_channels)
                
                dX[:, y:h, x:w, :] += d_out[:, y, x, :].dot(W_reshaped.T)\
                .reshape(batch_size, self.filter_size, self.filter_size, channels)
        
        # cut off the padding
        if self.padding:
            dX = dX[:, pad_width:height-pad_width, pad_width:width-pad_width, :]

        self.W.grad = dW
        self.B.grad = np.sum(d_out.reshape(-1, out_channels), axis=0)

        return dX
                

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()
        
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        out_shape = (batch_size, out_height, out_width, channels)
        out_data = np.zeros(out_shape)
        
        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size
                
                in_data = X[:, h_begin:h_end, w_begin:w_end, :]
                self.build_mask(x=in_data, pos=(y, x))
                out_data[:, y, x, :] = np.max(in_data, axis=(1, 2))
            
        return out_data
        

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                h_begin, w_begin = y * self.stride, x * self.stride
                h_end, w_end = h_begin + self.pool_size, w_begin + self.pool_size
                
                dX[:, h_begin:h_end, w_begin:w_end, :] += d_out[:, y:y + 1, x:x + 1, :] * self.masks[(y, x)]   
        return dX
    
    def build_mask(self, x, pos):
        mask = np.zeros_like(x)
        n, h, w, c = x.shape
        x = x.reshape(n, h * w, c)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((n, c))
        mask.reshape(n, h * w, c)[n_idx, idx, c_idx] = 1
        self.masks[pos] = mask

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
