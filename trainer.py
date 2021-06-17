import os

import numpy as np
import scipy.io

from SGD import SGD
from Softmax import Softmax
from forward_backward_pass import forward_pass, backward_pass
from nn_layer import NNLayer
from train_utils import calc_ce_loss, eval_train, eval_val, plot_acc_and_loss, shuffle_dataset


class NNTrain:
    def __init__(self, epochs, batch_size, lr, momentum, n_labels, dataset_name, n_layers, log_interval=50,
                 architecture=None, act_name='relu', subsample_ratio=1., plot_graphs=True, use_early_stop=False,
                 max_epochs_no_improve=50, l2_reg=0.0, lr_scheduler=None):
        """
        Class to build an train a Softmax regression
        :param epochs: number of epochs to run
        :param batch_size: size of batches for the SGD
        :param lr: learning rate
        :param momentum: momentum ratio for the SGD+Momentum optimizer
        :param n_labels: number of labels for the classification task
        :param n_layers: number of layers for the network
        :param log_interval: number of batch to print status
        :param architecture: network architecture widths (Optional), if not specified, will be determined automatically
        :param act_name: activation type to use in the NN layers
        :param subsample_ratio: ratio to subsample the dataset for accuracy calculation
        :param plot_graphs: boolean that indicates if we wish to plot accuracy and loss graphs
        :param use_early_stop: if this parameter is true, we will apply early stopping on the validation change
        :param max_epochs_no_improve: in case we use early stopping, this parameter defines the max number of epochs
        to wait with no improvement
        :param l2_reg: parameter to regularize the weight using the l2 norm. default is zero (meaning no regularization)
        :param lr_scheduler: learning rate scheduler to use for updating the lr. Default is no scheduling.
        """
        self.act_name = act_name
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.n_labels = n_labels
        self.dataset_name = dataset_name
        self.n_layers = n_layers
        self.log_interval = log_interval
        self.random_seed = 1234
        self.subsample_ratio = subsample_ratio
        self.plot_graphs = plot_graphs
        self.lr_scheduler = lr_scheduler

        # Default width for all layers if not specified by architecture
        self.default_width = 50

        # Early stopping parameters that holds best val_loss achieved and how many epochs the val loss did not improved
        self.use_early_stop = use_early_stop
        self.min_val_loss = np.Inf
        self.epochs_no_improve = 0
        self.max_epochs_no_improve = max_epochs_no_improve

        # Regularization
        self.l2_reg = l2_reg

    def create_net(self, input_feat_dim):
        """
        Create network neural network with n_layers
        :param input_feat_dim:
        :return: network architecture as a list of layers in a sequence with softmax at the end
        """
        assert self.architecture is None or self.n_layers == len(self.architecture)

        net = []
        if self.n_layers < 2:
            softmax = Softmax(dim=input_feat_dim, n_labels=self.n_labels)
            self.architecture = [input_feat_dim]
        else:
            if self.architecture is None:
                # Define default architecture in a case one was not specified
                self.architecture = [input_feat_dim] + [max(self.n_labels, int(self.default_width/(i+1))) for i in range(0, self.n_layers - 2)] + [self.n_labels]
                print(f'Network architecture: {self.architecture}')
            elif self.architecture[0] != input_feat_dim:
                print('First dimension is not equal to features dimension - Automatic fix is applied')
                self.architecture[0] = input_feat_dim
            for i in range(self.n_layers - 1):
                net.append(
                    NNLayer(dim_in=self.architecture[i], dim_out=self.architecture[i + 1], act_name=self.act_name))

            softmax = Softmax(dim=self.architecture[-1], n_labels=self.n_labels)

        return net, softmax

    def get_data(self):
        """
        Load dataset
        :return: returns the dataset specified by dataset_name
        """
        dataset_dict = None
        try:
            base_path = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
            dataset_dict = scipy.io.loadmat(f'{base_path}/data/{self.dataset_name}.mat')
            # shuffle val dataset - not mandatory
            per_v = np.random.permutation(dataset_dict['Yv'].shape[1])
            dataset_dict['Yv'] = dataset_dict['Yv'][:, per_v]
            dataset_dict['Cv'] = dataset_dict['Cv'][:, per_v]
        except Exception as e:
            print(f'No matching dataset path: data/{self.dataset_name}.mat')
        return dataset_dict

    def train(self):
        """
        Main training loop, which load the optimizer, the network and dataset.
        Then it start training the network for n_epochs. At the end of the training process we plot the accuracy
        and loss graphs for both train and validation datasets.
        :return: None
        """
        np.random.seed(self.random_seed)

        dataset_dict = self.get_data()  # Contains 'Ct', 'Cv', 'Yt', 'Yv'

        # Create optimizer
        optim = SGD(self.lr, self.momentum, self.l2_reg, self.lr_scheduler)

        # Create feed-forward network and softmax layer
        net, softmax = self.create_net(input_feat_dim=dataset_dict['Yt'].shape[0])

        train_loss_lst, val_loss_lst = [], []
        train_acc_lst, val_acc_lst = [], []

        for epoch in range(self.epochs):
            agg_loss = 0
            shuffle_dataset(dataset_dict)  # Shuffle dataset at every epoch
            for batch_idx, batch_start in enumerate(range(0, dataset_dict['Yt'].shape[1], self.batch_size)):
                Xt = dataset_dict['Yt'][:, batch_start:batch_start + self.batch_size]
                Yt = dataset_dict['Ct'][:, batch_start:batch_start + self.batch_size]

                # Forward pass
                sm_out = forward_pass(Xt, net, softmax)

                # Calculate loss
                loss = calc_ce_loss(sm_out, Yt)

                backward_pass(Yt, net, optim, sm_out, softmax)

                agg_loss += loss

                # print progress
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_start + self.batch_size, dataset_dict['Yt'].shape[1],
                        100. * batch_start / dataset_dict['Yt'].shape[1], agg_loss / (batch_idx + 1)))

            # Calculate train mean epoch loss and accuracy
            eval_train(agg_loss, batch_idx, dataset_dict, net, softmax, train_acc_lst, train_loss_lst,
                       self.subsample_ratio)
            # Calculate val loss and accuracy
            curr_val_loss = eval_val(dataset_dict, net, softmax, val_acc_lst, val_loss_lst, self.subsample_ratio)

            # Early stopping
            do_early_stop = self.early_stopping(curr_val_loss)
            if do_early_stop:
                break

            # Update learning rate
            optim.update_lr(epoch, self.epochs)

        if self.plot_graphs:
            plot_acc_and_loss(train_acc_lst, train_loss_lst, self.dataset_name, data_type='Train')
            plot_acc_and_loss(val_acc_lst, val_loss_lst, self.dataset_name, data_type='Validation')

        print(f'Final results: Train accuracy={round(train_acc_lst[-1], 2)}; Train loss={round(train_loss_lst[-1], 2)};'
              f' Validation Accuracy={round(val_acc_lst[-1], 2)}; Validation loss={round(val_loss_lst[-1], 2)}')

        return train_acc_lst, train_loss_lst, val_acc_lst, val_loss_lst

    def early_stopping(self, curr_val_loss):
        """
        Function that checks if we use early stopping, and if so checks if validation loss did not improve over a
        predefined number of epochs. If condition is met, we stop the run.
        :param curr_val_loss: the current calculated validation loss
        :return: True iff the conditions for the early stopping were met.
        """
        if self.use_early_stop:
            if curr_val_loss < self.min_val_loss:
                self.min_val_loss = curr_val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve == self.max_epochs_no_improve:
                print(f'Early stopping due to no improvement in validation loss for: {self.epochs_no_improve} epochs')
                return True
        return False


if __name__ == '__main__':
    # trainer = NNTrain(epochs=100, batch_size=64, lr=1e-3, momentum=0, n_labels=5, dataset_name='PeaksData',
    #                   n_layers=7, subsample_ratio=1, act_name='relu', plot_graphs=True, use_early_stop=False, l2_reg=0,
    #                   lr_scheduler='1cycle')

    trainer = NNTrain(epochs=100, batch_size=64, lr=1e-3, momentum=0.5, n_labels=5, dataset_name='GMMData',
                      n_layers=4, act_name='relu', subsample_ratio=1, plot_graphs=True, use_early_stop=False, l2_reg=1e-3,
                      lr_scheduler='1cycle')

    trainer.train()
