import time

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from trainer import NNTrain


def test_param_and_plot(param_to_iterate: list, param_name: str, trainer_params: dict):
    """
    Test and plot performance with respect to network length.
    :param param_to_iterate: different lengths to check
    :param param_name: parameter to tune
    :param trainer_params: parameters to initiate trainer
    :return: None
    """
    epochs_lst = [e + 1 for e in range(trainer_params['epochs'])]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    # colors = cm.rainbow(np.linspace(0, 1, len(param_to_iterate)))
    val_acc_lasts, val_loss_lasts = [], []

    for param_val in param_to_iterate:
        trainer_params.update({param_name: param_val, 'log_interval': 50, 'plot_graphs': False})
        trainer = NNTrain(**trainer_params)
        train_start_t = time.time()
        train_acc, train_loss, val_acc, val_loss = trainer.train()

        val_acc_lasts.append(val_acc[-1])
        val_loss_lasts.append(val_loss[-1])

        print(f'Train for {param_val} took: {time.time() - train_start_t} seconds')
        axes[0][0].plot(epochs_lst, train_acc, label=f'{param_name}={param_val}')
        axes[0][1].plot(epochs_lst, val_acc, label=f'{param_name}={param_val}')
        axes[1][0].plot(epochs_lst, train_loss, label=f'{param_name}={param_val}')
        axes[1][1].plot(epochs_lst, val_loss, label=f'{param_name}={param_val}')

    axes[0][0].set_xlabel('Epochs')
    axes[0][1].set_xlabel('Epochs')

    axes[0][0].set_ylabel('Train Accuracy')
    axes[0][1].set_ylabel('Val Accuracy')

    axes[0][0].set_title(f'Train Accuracy of different {param_name}')
    axes[0][1].set_title(f'Val Accuracy with different {param_name}')

    axes[0][0].legend()
    axes[0][1].legend()

    axes[1][0].set_xlabel('Epochs')
    axes[1][1].set_xlabel('Epochs')

    axes[1][0].set_ylabel('Train Loss')
    axes[1][1].set_ylabel('Val Loss')

    axes[1][0].set_title(f'Train Loss of different {param_name}')
    axes[1][1].set_title(f'Val Loss with different {param_name}')

    plt.suptitle(
        f'Accuracy and Loss with respect to the epochs with different {param_name} on the {trainer_params["dataset_name"]}')

    axes[1][0].legend()
    axes[1][1].legend()

    plt.show()

    print(f'Validation Accuracy: {val_acc_lasts}')
    print(f'Validation Loss: {val_loss_lasts}')

    # plot_acc_and_loss(val_acc_lasts, val_loss_lasts, trainer_params['dataset_name'], 'Validation', 'Number of Layers')


def full_hyper_parameter_tuning(dataset_name, n_labels):
    network_lengths = [6, 7, 9, 10]
    lrs = [1e-3, 5e-4]
    batch_sizes = [32, 64]
    momentum_values = [0, 0.05, 0.1, 0.2]
    l2_regs = [0.1, 0.01, 0.001, 0]
    lr_scheds = ['step_decay', '1cycle', None]

    results = pd.DataFrame(columns=['params', 'train_acc', 'train_loss', 'val_acc', 'val_loss'])

    print(f'Started full parameter tuning for dataset: {dataset_name} with: '
          f'{len(network_lengths) * len(lrs) * len(batch_sizes) * len(momentum_values) * len(l2_regs) * len(lr_scheds)} total combinations')

    curr_iter = 0

    for lr in tqdm(lrs):
        for n_layers in network_lengths:
            for batch_size in batch_sizes:
                for momentum in momentum_values:
                    for l2_reg in l2_regs:
                        for lr_scheduler in lr_scheds:
                            curr_iter += 1
                            print(f'Current Iteration #{curr_iter}')
                            trainer_params = {'lr': lr, 'n_layers': n_layers, 'batch_size': batch_size,
                                              'momentum': momentum, 'l2_reg': l2_reg, 'lr_scheduler': lr_scheduler,
                                              'epochs': 100, 'n_labels': n_labels, 'dataset_name': dataset_name,
                                              'act_name': 'tanh', 'subsample_ratio': 1, 'plot_graphs': False}
                            trainer = NNTrain(**trainer_params)
                            train_acc, train_loss, val_acc, val_loss = trainer.train()
                            results = results.append([{'params': trainer_params, 'train_acc': train_acc,
                                                       'train_loss': train_loss, 'val_acc': val_acc,
                                                       'val_loss': val_loss}])

    results.to_csv('peaks_parameter_tuning_res.csv', index=False)


if __name__ == '__main__':
    # Test and plot performance with respect to different lengths
    # trainer_params = {'epochs': 100, 'batch_size': 64, 'lr': 1e-3, 'momentum': 0, 'n_labels': 5,
    #                   'dataset_name': 'PeaksData', 'subsample_ratio': 1, 'act_name': 'relu'}
    # test_param_and_plot(param_to_iterate=[i + 1 for i in range(10)], param_name='n_layers',
    #                     trainer_params=trainer_params)
    #
    # Test learning rate
    # trainer_params = {'epochs': 100, 'batch_size': 64, 'momentum': 0, 'n_labels': 5, 'act_name': 'relu',
    #                   'dataset_name': 'GMMData', 'subsample_ratio': 1, 'n_layers': 4}
    # test_param_and_plot(param_to_iterate=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5], param_name='lr',
    #                     trainer_params=trainer_params)

    # Test batch size
    # peaks_params = {'epochs': 100, 'lr': 1e-3, 'momentum': 0, 'n_labels': 5, 'act_name': 'relu',
    #                 'dataset_name': 'PeaksData', 'subsample_ratio': 1, 'n_layers': 7}
    # gmm_params = {'epochs': 100, 'lr': 1e-3, 'momentum': 0, 'n_labels': 5, 'act_name': 'relu',
    #               'dataset_name': 'GMMData', 'subsample_ratio': 1, 'n_layers': 4}
    # test_param_and_plot(param_to_iterate=[16, 32, 64, 128, 256], param_name='batch_size',
    #                     trainer_params=gmm_params)

    # Test momentum
    # peaks_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'act_name': 'relu',
    #                 'dataset_name': 'PeaksData', 'subsample_ratio': 1, 'n_layers': 7}
    # gmm_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'act_name': 'relu',
    #                       'dataset_name': 'GMMData', 'subsample_ratio': 1, 'n_layers': 4}
    # test_param_and_plot(param_to_iterate=[0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], param_name='momentum',
    #                     trainer_params=gmm_params)

    # Test l2 regularization
    # peaks_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'momentum': 0, 'act_name': 'relu',
    #                 'dataset_name': 'PeaksData', 'subsample_ratio': 1, 'n_layers': 7}
    # gmm_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'momentum': 0.5, 'act_name': 'relu',
    #               'dataset_name': 'GMMData', 'subsample_ratio': 1, 'n_layers': 4}
    # test_param_and_plot(param_to_iterate=[1, 0.1, 0.01, 0.001, 0], param_name='l2_reg',
    #                     trainer_params=gmm_params)

    # Test lr scheduler
    # gmm_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'momentum': 0.5, 'act_name': 'relu',
    #               'dataset_name': 'GMMData', 'subsample_ratio': 1, 'n_layers': 4, 'l2_reg': 1e-3}
    # peaks_params = {'epochs': 100, 'lr': 1e-3, 'batch_size': 64, 'n_labels': 5, 'momentum': 0, 'act_name': 'relu',
    #                 'dataset_name': 'PeaksData', 'subsample_ratio': 1, 'n_layers': 7, 'l2_reg': 1e-3}
    # test_param_and_plot(param_to_iterate=['step_decay', '1cycle', None], param_name='lr_scheduler',
    #                     trainer_params=gmm_params)
    #
    full_hyper_parameter_tuning(dataset_name='PeaksData', n_labels=5)
    # df = pd.read_csv('peaks_parameter_tuning_res.csv')
    # df.head()
