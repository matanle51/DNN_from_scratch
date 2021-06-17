import copy
import random

import numpy as np

from SGD import SGD
from forward_backward_pass import forward_pass, backward_pass
from grad_test import get_eps_list, plot_grad_test, get_onehot_enc
from trainer import NNTrain
from train_utils import calc_ce_loss, plot_logscale_diff


def grad_test_net(trainer, n_iter, eps0=1.):
    dataset_dict = trainer.get_data()  # Contains 'Ct', 'Cv', 'Yt', 'Yv'
    # Xt = dataset_dict['Yt'][:, 0:trainer.batch_size]
    # Yt = dataset_dict['Ct'][:, 0:trainer.batch_size]

    n_samples, n_feat, n_labels = 1, 2, 5
    Xt = np.random.randn(n_feat, n_samples)
    y = [0] * int(n_samples / 2) + [1] * int(n_samples / 2)
    random.shuffle(y)
    Yt = get_onehot_enc([0], n_labels).T

    # Create optimizer and NN
    optim = SGD(lr=trainer.lr, momentum=0)
    net, softmax = trainer.create_net(input_feat_dim=dataset_dict['Yt'].shape[0])

    # Save clean copy of net since the backward pass updates the net weights
    net_orig, softmax_orig = copy.deepcopy(net), copy.deepcopy(softmax)

    # Create random vector d for each layer
    dw_lst = [np.random.randn(trainer.architecture[i], trainer.architecture[i + 1]) for i in range(trainer.n_layers - 1)] + [np.random.randn(trainer.architecture[-1], trainer.n_labels)]
    db_lst = [np.random.randn(trainer.architecture[i + 1], 1) for i in range(trainer.n_layers - 1)] + [np.random.randn(trainer.n_labels, 1)]
    all_d_lst = [np.concatenate((db.T, dw), axis=0) for dw, db in zip(dw_lst, db_lst)]
    all_d_concat = np.concatenate(all_d_lst, axis=0).flatten()

    # Calc forward and backward for original network
    sm_out = forward_pass(Xt, net, softmax)

    # Calculate backward pass and concat all gradients together
    grads = backward_pass(Yt, net, optim, sm_out, softmax, mode='grad_ver')
    all_grads_lst = [np.concatenate((db.T, dw.T), axis=0) for dw, db in grads][::-1]  # Concatenate and reverse the grads (order is reversed due to the back-prop)
    all_grads_concat = np.concatenate(all_grads_lst, axis=0).flatten()

    fx = calc_ce_loss(sm_out, Yt)

    diff_f_lst, diff_f_g_lst = [], []
    all_eps = get_eps_list(eps0, n_iter)

    for eps in all_eps:
        # Create another copy for use of nn and SGD objects copy to not manipulate weights twice
        net_c, softmax_c = copy.deepcopy(net_orig), copy.deepcopy(softmax_orig)

        # Add perturbation to all layers of the network
        for i, layer in enumerate(net_c):
            layer.update_w(layer.W + eps * dw_lst[i].T, layer.b + eps * db_lst[i])  # Update the copy net weights
        softmax_c.update_w(softmax_c.W + eps * dw_lst[-1].T, softmax_c.b + eps * db_lst[-1])

        # Calc f(x+ed)
        sm_out = forward_pass(Xt, net_c, softmax_c)

        # calc loss for f(x+eps*d)
        fx_ed = calc_ce_loss(sm_out, Yt)

        # Calc gradient
        diff_f = abs(fx_ed - fx)
        diff_f_g = abs(fx_ed - fx - eps * all_d_concat.T.dot(all_grads_concat))

        diff_f_lst.append(diff_f)
        diff_f_g_lst.append(diff_f_g)

    plot_grad_test(all_eps, diff_f_g_lst, diff_f_lst, test_type='weights for full network gradient')
    plot_logscale_diff(diff_f_lst, diff_f_g_lst, test_type='weights for full network gradient')


if __name__ == '__main__':
    trainer = NNTrain(epochs=1, batch_size=1, lr=1e-3,momentum=0, n_labels=5,
                      dataset_name='PeaksData',  act_name='tanh', n_layers=3, architecture=[5, 5, 5])

    grad_test_net(trainer, n_iter=20, eps0=0.05)
