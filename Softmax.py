import numpy as np


class Softmax:
    """
    A class that combines the last linear layer W^TX+b with the softmax objective
    """
    def __init__(self, dim, n_labels):
        self.n_labels = n_labels
        self.W = np.random.randn(n_labels, dim).astype(np.float64) * np.sqrt(2/dim)  # We hold W as W.T for calc
        # self.W = np.random.randn(dim, n_labels).astype(np.float64)
        self.b = np.random.randn(n_labels, 1).astype(np.float64)

        # Velocity
        self.vw = np.zeros(self.W.shape)
        self.vb = np.zeros(self.b.shape)

        # Current input
        self.curr_input = None

    def forward(self, X: np.array):
        """
        net_out is X^Tw + b is the Softmax input, which is the output from the last layer.
        We wish to calculate the softmax regression: exp(X^Tw_k)/{sum_j(exp(X^Twj)}
        :param X: input for forward pass
        :return: The softmax result: softmax(X^Tw + b)
        """
        self.curr_input = X
        net_out = self.W.dot(X) + self.b
        # net_out = self.W.T.dot(X) + self.b
        softmax_res = self.softmax_function(net_out)

        return softmax_res

    def softmax_function(self, net_out: np.array):
        """
        Softmax function calculation
        :param net_out: The output of the last layer := X^Tw + b
        :return: Softmax calculation result
        """
        # For safety, XW + b is not bounded, and can be computed as Inf,
        # Therefore to prevent calculation to be performed without overflow, we reduce: max_j{x^tw_j}
        net_out -= np.max(net_out, axis=0)
        softmax_res = (np.exp(net_out.T) / np.reshape(np.sum(np.exp(net_out), axis=0), (-1, 1))).T
        return softmax_res

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

    def backward_w(self, sm_out: np.array, y: np.array):
        """
        The backward will compute the gradient of the loss with respect to the *weights (and biases)*.
        delta_xF = 1/m * W * [exp(W^TX)/(sum_j{exp(wj^TX)}) - C] in R^(n*m).
        where W in R^(n*l) and C in R^(l*m). l is the number of labels, m is the number of samples
        :param sm_out: softmax output result from the forward pass
        :param y: The labels of the current batch
        :return: derivatives with respect to the weights and biases -> dW and db
        """
        delta = sm_out - y

        # dw = np.dot(self.curr_input, delta.T)
        # dw = np.dot(self.curr_input, delta.T).T
        dw = delta.dot(self.curr_input.T)  # we output dw.T since W is also transposed and we need to add dw to W
        db = np.sum(delta, axis=1).reshape(-1, 1)

        return dw, db

    def backward_x(self, sm_out: np.array, y: np.array):
        """
        The backward will compute the gradient of the loss with respect to the *data*.
        delta_xF = 1/m * X * [exp(W^TX)/(sum_j{exp(wj^TX)}) - C] in R^(n*m).
        where W in R^(n*l) and C in R^(l*m). l is the number of labels, m is the number of samples.
        :param sm_out: softmax output result from the forward pass
        :param y: The labels of the current batch
        :return: derivatives with respect to the data X
        """
        delta = sm_out - y
        dx = np.dot(self.W.T, delta)
        # dx = np.dot(self.W, delta)

        return dx
