import matplotlib.pyplot as plt
import numpy as np

from Softmax import Softmax
from forward_backward_pass import forward_pass
from matplotlib.ticker import MaxNLocator

eps = 2e-16  # To avoid log on zero


def calc_ce_loss(softmax_out: np.array, y: np.array):
    """
    Function to calculate the cross entropy loss. Note the we return the mean value of the loss
    :param softmax_out: softmax output results from the network softmax layer
    :param y: batch labels
    :return: mean cross entropy loss
    """
    ce = -np.log(softmax_out + eps) * y
    return np.mean(np.max(ce, axis=0))


def calc_accuracy(sm_out: np.array, Y: np.array):
    """
    Calculate the accuracy of the network, by comaring the class of the max value from the softmax to the true label
    of every input x in X
    :param sm_out: softmax result
    :param Y: batch labels
    :return: accuracy %
    """
    n_correct = np.sum(np.argmax(Y, 0) == np.argmax(sm_out, 0))  # sum number of correct classifications
    return round((n_correct / Y.shape[1]) * 100, 2)


def plot_acc_and_loss(acc_lst: list, loss_lst: list, dataset_name: str, data_type: str, xlabel_title: str = 'Epoch'):
    """
    Plot accuracy and loss graphs for the training process
    :param acc_lst: accuracy list calculated from all epochs
    :param loss_lst: loss list calculated from all epochs
    :param dataset_name: name of the dataset we plot results for
    :param data_type: type of the data we use (train or validation)
    :param xlabel_title: label name for x-axis
    :return: None
    """
    epochs_lst = [i for i in range(len(acc_lst))]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(xlabel_title)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs_lst, acc_lst, color='r', linestyle='-', label='Accuracy')  # , marker='X'
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_lst, loss_lst, color=color, linestyle='-', label='Loss')  # , marker='X'
    ax2.tick_params(axis='y', labelcolor=color)

    plt.suptitle(f'{dataset_name} - {data_type} Accuracy and Loss')
    plt.figlegend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


def plot_logscale_diff(diff1, diff2, test_type):
    X = [i+1 for i in range(len(diff1))]

    # Assign variables to the y axis part of the curve
    diff1 = np.log2(diff1)
    diff2 = np.log2(diff2)

    # Plotting both the curves simultaneously
    plt.plot(X, diff1, color='r', label='|f(x+ed)-f(x)|')
    plt.plot(X, diff2, color='g', label='|f(x+ed)-f(x)-ed*grad|')

    plt.xlabel("Test Iteration")
    plt.xticks(X)
    plt.ylabel("Difference value")
    plt.title(f"Log-scale gradient test graphs w.r.t {test_type}")

    plt.legend()
    plt.show()


def eval_train(agg_loss: float, batch_idx: int, dataset_dict: dict, net: list, softmax: Softmax, train_acc_lst: list,
               train_loss_lst: list, subsample_ratio=1):
    """
    Evaluate training loss and accuracy. Note - there is no need to shuffle Yt and Ct since they were already shuffled
    :param agg_loss: aggregated train epoch loss
    :param batch_idx: index to the current batch
    :param dataset_dict: dataset
    :param net: network layers
    :param softmax: softmax class
    :param train_acc_lst: list of accuracy values at each epoch
    :param train_loss_lst: list of loss values at each epoch
    :param subsample_ratio: ratio for subsampling the datasets
    :return: None
    """
    n_subsample = int(dataset_dict['Yt'].shape[1] * subsample_ratio)
    train_loss_lst.append(agg_loss / (batch_idx + 1))
    Xt, Yt = dataset_dict['Yt'][:, :n_subsample], dataset_dict['Ct'][:, :n_subsample]
    train_sm_out = forward_pass(Xt, net, softmax)
    train_acc_lst.append(calc_accuracy(train_sm_out, Yt))


def eval_val(dataset_dict: dict, net: list, softmax: Softmax, val_acc_lst: list, val_loss_lst: list,
             subsample_ratio: float = 1.0):
    """
    Evaluate validation loss and accuracy.
    :param dataset_dict: dataset
    :param net: network layers
    :param softmax: softmax class
    :param val_acc_lst: list of accuracy values at each epoch
    :param val_loss_lst: list of loss values at each epoch
    :param subsample_ratio: ratio for subsampling the datasets
    :return: Validation loss
    """
    n_subsample = int(dataset_dict['Yv'].shape[1] * subsample_ratio)
    Xv, Yv = dataset_dict['Yv'][:, :n_subsample], dataset_dict['Cv'][:, :n_subsample]
    val_sm_out = forward_pass(Xv, net, softmax)
    val_loss = calc_ce_loss(val_sm_out, Yv)
    val_loss_lst.append(val_loss)
    val_accuracy = calc_accuracy(val_sm_out, Yv)
    val_acc_lst.append(val_accuracy)
    print('Val: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(val_loss, val_accuracy))

    return val_loss


def shuffle_dataset(dataset_dict: dict):
    """
    We shuffle the dataset in the beginning of every epoch in order to give the network different batches.
    This should help the network generalize better since the batches are constantly changing between epochs.
    The validation data is shuffled so that randomly subsample could be applied in evaluation.
    :param dataset_dict: dataset
    :return: None
    """
    # shuffle train dataset to get mini-batches of all classes
    per_t = np.random.permutation(dataset_dict['Yt'].shape[1])
    dataset_dict['Yt'] = dataset_dict['Yt'][:, per_t]
    dataset_dict['Ct'] = dataset_dict['Ct'][:, per_t]
    # Shuffle test for later use
    per_v = np.random.permutation(dataset_dict['Yv'].shape[1])
    dataset_dict['Yv'] = dataset_dict['Yv'][:, per_v]
    dataset_dict['Cv'] = dataset_dict['Cv'][:, per_v]
