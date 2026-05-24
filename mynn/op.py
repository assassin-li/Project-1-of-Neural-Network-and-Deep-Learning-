from abc import abstractmethod
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=None, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is None:
            # He initialization for ReLU
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(0, std, size=(in_dim, out_dim))
            self.b = np.zeros((1, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
            self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return self.input @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = (self.input).T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        in_dim = in_channels*kernel_size*kernel_size
        if initialize_method is None:
            # He initialization for ReLU
            std = np.sqrt(2.0 / in_dim)
            self.W = np.random.normal(0, std, size=(out_channels,in_channels,kernel_size,kernel_size))
            self.b = np.zeros((out_channels))
        else:
            self.W = initialize_method(size=(out_channels,in_channels,kernel_size,kernel_size))
            self.b = initialize_method(size=(out_channels,))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}
        

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, in_channels, H, W]
        W : [out_channels, in_channels, k, k]
        """
        self.input = X
        batch, input_channel, H_in, W_in = X.shape
        assert input_channel == self.in_channel, f"input_channel doesn't match model assumption, expected {self.in_channel}, received {input_channel}"
        K = self.kernel_size

        H_out = (H_in - K + 2 * self.padding) // self.stride + 1
        W_out = (W_in - K + 2 * self.padding) // self.stride + 1

        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            X_pad = X

        windows = sliding_window_view(X_pad, (K, K), axis=(2, 3))
        if self.stride > 1:
            windows = windows[:, :, ::self.stride, ::self.stride, :, :]

        output = np.einsum('bchwij, ocij -> bohw', windows, self.W) + self.b[None, :, None, None]
        self.windows = windows
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, H_out, W_out]
        """
        X = self.input
        batch, input_channel, H_in, W_in = X.shape
        K = self.kernel_size
        _, out_channel, H_out, W_out = grads.shape

        self.grads['W'] = np.einsum('bohw, bchwij -> ocij', grads, self.windows)
        self.grads['b'] = np.einsum('bohw -> o', grads)

        if self.padding > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            dX_pad = np.zeros_like(X_pad)
        else:
            dX_pad = np.zeros_like(X)

        for i in range(K):
            for j in range(K):
                contrib = np.einsum('bohw, oc -> bchw', grads, self.W[:, :, i, j])
                dX_pad[:, :, i:i+H_out*self.stride:self.stride, j:j+W_out*self.stride:self.stride] += contrib

        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Flatten(Layer):
    """
    Flatten layer to bridge conv2D output [batch, C, H, W] to Linear input [batch, C*H*W].
    """
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        if X.size == 0:
            return np.empty((0, int(np.prod(X.shape[1:]))))
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        if grads.size == 0:
            return np.empty(self.input_shape)
        return grads.reshape(self.input_shape)


class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.probs= None
        self.label = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        predicts = softmax(predicts)
        self.probs= predicts
        self.label = labels
        
        # calculate loss
        class_num = self.probs.shape[0]
        right_prob = self.probs[np.arange(class_num),self.label]
        loss = - np.mean(np.log(right_prob + 1e-8))
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        class_num = self.probs.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(class_num), self.label] -= 1
        self.grads = self.grads/class_num
        self.model.backward(self.grads)
        return self.grads

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition