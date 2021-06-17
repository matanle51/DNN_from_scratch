import copy

import matplotlib.pyplot as plt
import numpy as np

from nn_layer import NNLayer


def get_eps_list(eps0, n_iter):
    all_eps = [0.5 ** i * eps0 for i in range(0, n_iter)]
    return all_eps


def Jx_mV(nn_layer, v):
    """
    For testing - we compute the multiplication of the matrix from the NNLayer with a vector (without the Transpose).
    :param nn_layer: network for which we use the weights
    :param v: vector to multiply with
    :return: Jac_x times v
    """
    f_out = nn_layer.W.dot(nn_layer.curr_input) + nn_layer.b
    act_d = nn_layer.get_activation_derivative(f_out)

    Jx = np.diag(act_d.flatten()).dot(nn_layer.W)
    return Jx.dot(v)


def jac_test_X(X, n_iter=10, eps0=0.1):
    n_feat = X.shape[0]

    # Define one weights layer + softmax
    nn_layer = NNLayer(n_feat, n_feat, act_name='tanh')
    nn_layer_copy = copy.deepcopy(nn_layer)  # Since we update the class during the calculation, we need a duplicate

    # Generate random Vector d with ||d||= O(1)
    dx = np.random.randn(n_feat, X.shape[1])

    diff_f_lst, diff_f_g_lst = [], []
    all_eps = get_eps_list(eps0, n_iter)

    for eps in all_eps:
        # Calc f(x)
        fx = nn_layer.forward(X)

        # Create a copy of the original softmax class with perturbation weights and calc f(x+e*d)
        fx_ed = nn_layer_copy.forward(X + eps*dx)

        # Calc jac with respect to the data (X)
        jac_x = Jx_mV(nn_layer, eps*dx)

        diff_f = np.linalg.norm(fx_ed - fx, ord=2)
        diff_f_g = np.linalg.norm(fx_ed - fx - jac_x, ord=2)

        diff_f_lst.append(diff_f)
        diff_f_g_lst.append(diff_f_g)

    plot_jac_test(all_eps, diff_f_g_lst, diff_f_lst, test_type='X')


def test_transpose(X):
    n_feat = X.shape[0]

    # Define one weights layer + softmax
    nn_layer = NNLayer(n_feat, n_feat, act_name='tanh')

    # Generate random Vector d with ||d||= O(1)
    v = np.random.randn(n_feat, X.shape[1])
    u = np.random.randn(n_feat, X.shape[1])

    # Update layer class with current X
    nn_layer.curr_input = X

    # Calculate Jx times V
    JxV = Jx_mV(nn_layer, v)

    # Calculate Jx^T time V
    JxtV = nn_layer.backward_x(u)

    delta = np.abs(u.T.dot(JxV) - v.T.dot(JxtV))

    print(f'Transpose delta test: {delta.item()}')

    return delta


def plot_jac_test(all_eps, diff_f_g_lst, diff_f_lst, test_type):
    # Plot the diff functions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axes[0].plot(all_eps, diff_f_lst, 'r', label='||f(x+ed)-f(x)||')
    axes[0].set_xlim([all_eps[0], 0])
    axes[0].set_xlabel('Epsilon value')
    axes[0].set_ylabel('l2 Norm difference value')
    axes[0].legend()

    axes[1].plot(all_eps, diff_f_g_lst, 'b', label='||f(x+ed)-f(x)-JacMV(x,ed)||')
    axes[1].set_xlim([all_eps[0], 0])
    axes[1].set_xlabel('Epsilon value')
    axes[1].set_ylabel('l2 Norm difference value')
    axes[1].legend()
    plt.suptitle(f'JacMV test results with respect to {test_type}')
    plt.show()

    print('=============== |f(x+ed)-f(x)| ====================')
    print(f'diff values : {diff_f_lst}')
    print(f'decrease ratio: {[round(diff_f_lst[i]/diff_f_lst[i+1],2) for i in range(0,len(diff_f_lst)-1)]}')
    print('===================================================\n')

    print('================ |f(x+ed)-f(x)-JacMV(x,ed)| ===================')
    print(f'diff values : {diff_f_g_lst}')
    print(f'decrease ratio: {[round(diff_f_g_lst[i]/diff_f_g_lst[i+1],2) for i in range(0,len(diff_f_g_lst)-1)]}')
    print('===========================================================')


if __name__ == '__main__':
    n_samples = 1
    n_feat = 5
    n_labels = 3

    X = np.random.randn(n_feat, n_samples)

    # Run jac test with respect to the data
    jac_test_X(X, n_iter=20)

    # Run transpose test
    deltas = []
    for i in range(100):
        deltas.append(test_transpose(X))
    print(f'Mean value of transpose test: {np.mean(deltas)}')