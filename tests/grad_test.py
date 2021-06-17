import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from Softmax import Softmax
from train_utils import plot_logscale_diff


def calc_loss(softmax, X, y):
    sm_out = softmax.forward(X)
    ce = -np.log(sm_out) * y
    return np.mean(np.max(ce, axis=0))


def get_eps_list(eps0, n_iter):
    all_eps = [0.5 ** i * eps0 for i in range(0, n_iter)]
    return all_eps


def grad_test_X(X, y, n_iter=10, eps0=0.1):
    n_labels = y.shape[0]
    n_feat, n_samples = X.shape[0], X.shape[1]

    # Define one weights layer + softmax
    softmax = Softmax(n_feat, n_labels)

    # Generate random Vector d with ||d||= O(1)
    dx = np.random.randn(n_feat, X.shape[1])

    diff_f_lst, diff_f_g_lst = [], []
    all_eps = get_eps_list(eps0, n_iter)

    for eps in all_eps:
        # Calc f(x)
        fx = calc_loss(softmax, X, y)

        # Create a copy of the original softmax class with perturbation weights and calc f(x+e*d)
        fx_ed = calc_loss(softmax, X + eps*dx, y)

        # Calc gradient with respect to the data (X)
        sm_out = softmax.forward(X)
        grad_x = softmax.backward_x(sm_out, y)

        diff_f = abs(fx_ed - fx)
        diff_f_g = abs(fx_ed - fx - eps * dx.T.dot(grad_x).item())

        diff_f_lst.append(diff_f)
        diff_f_g_lst.append(diff_f_g)

    plot_grad_test(all_eps, diff_f_g_lst, diff_f_lst, test_type='X')
    plot_logscale_diff(diff_f_lst, diff_f_g_lst, test_type='X')


def grad_test_weights(X, y, n_iter=10, eps0=0.1):
    n_labels = y.shape[0]
    n_feat = X.shape[0]

    # Define one weights layer + softmax
    softmax_orig = Softmax(n_feat, n_labels)

    # Generate random Vector d with ||d||= O(1)
    dw = np.random.randn(n_feat, n_labels)
    db = np.random.randn(n_labels, 1)
    d = np.concatenate((db.T, dw), axis=0).flatten()

    diff_f_lst, diff_f_g_lst = [], []
    all_eps = get_eps_list(eps0, n_iter)

    for eps in all_eps:
        # Calc f(x)
        softmax = copy.deepcopy(softmax_orig)
        fx = calc_loss(softmax, X, y)

        # Create a copy of the original softmax class with perturbation weights and calc f(x+e*d)
        softmax_d = copy.deepcopy(softmax_orig)
        softmax_d.update_w(softmax_d.W + eps*dw.T, softmax_d.b + eps*db)  # Update the softmax class copy weights
        fx_ed = calc_loss(softmax_d, X, y)

        # Calc gradient with respect to the weights
        sm_out = softmax.forward(X)
        grad_w, grad_b = softmax.backward_w(sm_out, y)
        grad = np.concatenate((grad_b.T, grad_w.T), axis=0).flatten()

        diff_f = abs(fx_ed - fx)
        diff_f_g = abs(fx_ed - fx - eps * d.T.dot(grad))

        diff_f_lst.append(diff_f)
        diff_f_g_lst.append(diff_f_g)

    plot_grad_test(all_eps, diff_f_g_lst, diff_f_lst, test_type='weights')
    plot_logscale_diff(diff_f_lst, diff_f_g_lst, test_type='weights')


def plot_grad_test(all_eps, diff_f_g_lst, diff_f_lst, test_type):
    # Plot the diff functions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    axes[0].plot(all_eps, diff_f_lst, 'r', label='|f(x+ed)-f(x)|')
    axes[0].set_xlim([all_eps[0], 0])
    axes[0].set_xlabel('Epsilon value')
    axes[0].set_ylabel('Difference value')

    axes[0].legend()

    axes[1].plot(all_eps, diff_f_g_lst, 'b', label='|f(x+ed)-f(x)-ed*grad|')
    axes[1].set_xlim([all_eps[0], 0])
    axes[1].set_xlabel('Epsilon value')
    axes[1].set_ylabel('Difference value')
    axes[1].legend()
    plt.suptitle(f'Gradient test results with respect to {test_type}')
    plt.show()

    print('=============== |f(x+ed)-f(x)| ====================')
    print(f'diff values : {diff_f_lst}')
    print(f'decrease ratio: {[round(diff_f_lst[i]/diff_f_lst[i+1],2) for i in range(0,len(diff_f_lst)-1)]}')
    print('===================================================\n')

    print('================ |f(x+ed)-f(x)-ed*grad| ===================')
    print(f'diff values : {diff_f_g_lst}')
    print(f'decrease ratio: {[round(diff_f_g_lst[i]/diff_f_g_lst[i+1],2) for i in range(0,len(diff_f_g_lst)-1)]}')
    print('===========================================================')


def get_onehot_enc(y, n_labels):
    """
    y are the samples' labels.
    """
    one_hot_enc = np.zeros((len(y), n_labels))
    one_hot_enc[np.arange(len(y)), y] = 1

    return one_hot_enc


if __name__ == '__main__':
    n_samples = 1
    n_feat = 5
    n_labels = 3

    X = np.random.randn(n_feat, n_samples)
    y = [0] * int(n_samples / 2) + [1] * int(n_samples / 2)
    random.shuffle(y)
    y_onehot = get_onehot_enc([0], n_labels).T

    # Run gradient test with respect to the weights
    grad_test_weights(X, y_onehot, n_iter=20)

    # Run gradient test with respect to the data
    grad_test_X(X, y_onehot, n_iter=20)

