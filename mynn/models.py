from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, lambda_list=None):
        self.lambda_list = lambda_list
        self.arch = [
            {'type': 'conv2D', 'in_channels': 1, 'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'conv2D', 'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'type': 'ReLU'},
            {'type': 'Flatten'},
            {'type': 'Linear', 'in_dim': 3136, 'out_dim': 32},
            {'type': 'ReLU'},
            {'type': 'Linear', 'in_dim': 32, 'out_dim': 10},
        ]
        self.layers = []
        optimizable_idx = 0
        for cfg in self.arch:
            if cfg['type'] == 'conv2D':
                layer = conv2D(
                    in_channels=cfg['in_channels'],
                    out_channels=cfg['out_channels'],
                    kernel_size=cfg['kernel_size'],
                    stride=cfg['stride'],
                    padding=cfg['padding'],
                    initialize_method=None
                )
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[optimizable_idx]
                    optimizable_idx += 1
                self.layers.append(layer)
            elif cfg['type'] == 'Linear':
                layer = Linear(in_dim=cfg['in_dim'], out_dim=cfg['out_dim'])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[optimizable_idx]
                    optimizable_idx += 1
                self.layers.append(layer)
            elif cfg['type'] == 'ReLU':
                self.layers.append(ReLU())
            elif cfg['type'] == 'Flatten':
                self.layers.append(Flatten())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list_path):
        with open(param_list_path, 'rb') as f:
            param_list = pickle.load(f)
        self.arch = param_list[0]
        self.lambda_list = None

        self.layers = []
        optimizable_idx = 0
        for cfg in self.arch:
            if cfg['type'] == 'conv2D':
                layer = conv2D(
                    in_channels=cfg['in_channels'],
                    out_channels=cfg['out_channels'],
                    kernel_size=cfg['kernel_size'],
                    stride=cfg['stride'],
                    padding=cfg['padding'],
                    initialize_method=None
                )
                self.layers.append(layer)
            elif cfg['type'] == 'Linear':
                layer = Linear(in_dim=cfg['in_dim'], out_dim=cfg['out_dim'])
                self.layers.append(layer)
            elif cfg['type'] == 'ReLU':
                self.layers.append(ReLU())
            elif cfg['type'] == 'Flatten':
                self.layers.append(Flatten())

        param_idx = 1
        for layer in self.layers:
            if layer.optimizable:
                layer.W = param_list[param_idx]['W']
                layer.b = param_list[param_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[param_idx]['weight_decay']
                layer.weight_decay_lambda = param_list[param_idx]['lambda']
                param_idx += 1

    def save_model(self, save_path):
        param_list = [self.arch]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)