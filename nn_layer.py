import numpy as np

from activations import relu, tanh


class NNLayer:
    def __init__(self, dim_in, dim_out, act_name='relu'):
        """
        Create and initialize weights
        :param dim_in: dimension in
        :type dim_out: dimension out
        :param act_name: activation type to use (currently supported are ReLU and tanh)
        """
        act_divide = 2 if act_name == 'relu' else 1  # Based on He initialization where in relu we need additional factor of 2 for relu since in average half of the neurons are dead (zeroed out) --> Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
        # Weights and biases
        self.W = np.random.randn(dim_out, dim_in).astype(np.float64) * np.sqrt(act_divide/dim_in)  # We hold W.T for calc
        self.b = np.random.randn(dim_out, 1).astype(np.float64)

        # Activation function
        self.act_name = act_name
        self.activation = self.get_activation(act_name)

        # Velocity
        self.vw = np.zeros(self.W.shape)
        self.vb = np.zeros(self.b.shape)

        # Current input
        self.curr_input = None

    def get_activation(self, act_name: str):
        """
        Retrieve activation function by name
        :param act_name: activation function name
        :return: activation function (currently relu ot tanh)
        """
        if act_name == 'relu':
            return relu
        elif act_name == 'tanh':
            return tanh
        else:
            raise NotImplementedError(f'Requested activation is not implemented: {act_name}')

    def get_activation_derivative(self, f_out: np.array):
        """
        Compute the activation derivative on the network forward output Wx+b
        :param f_out: Wx+b
        :return: activation derivative
        """
        if self.act_name == 'relu':
            act_d = np.where(f_out <= 0, 0, 1)
        elif self.act_name == 'tanh':
            act_d = 1 - self.activation(f_out) ** 2  # tanh(in)' = 1 - tanh(in)^2
        else:
            raise NotImplementedError(f'Activation function is not implememnted: {self.act_name}')
        return act_d

    def forward(self, X: np.array):
        """
        calculate the forward pass of the layer
        :param X: X in n*m where n is the number of features and m is the number of examples (mini-batch size in SGD)
        :return: The forward pass output W^TX+b
        """
        self.curr_input = X
        return self.activation(self.W.dot(X) + self.b)

    def backward_w(self, v: np.array):
        """
        Calculate the backward stage - Jac^T times vector v for the weights and biases
        :param v: vector from the backpropagation of layer after the current layer
        :return: Jw^T*v and Jb^T*v
        """
        f_out = self.W.dot(self.curr_input) + self.b
        act_d = self.get_activation_derivative(f_out)

        # Derivative of f_out transopse with respect to b times vector v
        JbtV = np.mean(act_d * v, axis=1).reshape(-1, 1)

        # Derivative of f_out traspose with respect to W times vector v
        JWtV = (act_d * v).dot(self.curr_input.T)

        return JWtV, JbtV

    def backward_x(self, v: np.array):
        """
        Calculate the backward stage - Jac^T times vector v with respect to the data
        :param v: vector from the backpropagation of layer after the current layer
        :return: Jx^T*v
        """
        f_out = self.W.dot(self.curr_input) + self.b
        act_d = self.get_activation_derivative(f_out)

        JXtV = self.W.T.dot(act_d * v)

        return JXtV

    def update_w(self, new_W: np.array, new_b: np.array, new_vw: np.array = None, new_vb: np.array = None):
        """
        Update the weights and biases with dw*eps and db*eps
        :param new_W: updated weights matrix
        :param new_b: updated bias vector
        :param new_vw: updated velocity of W
        :param new_vw: updated velocity of b
        :return: None
        """
        self.W = new_W
        self.b = new_b
        if new_vw is not None and new_vb is not None:
            self.vw = new_vw
            self.vb = new_vb
