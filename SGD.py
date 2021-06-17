import copy
import math


class SGD:
    def __init__(self, lr=1e-4, momentum=0.9, l2_reg=0.0, lr_scheduler=None):
        self.lr = lr
        self.momentum = momentum
        self.V = None
        self.l2_reg = l2_reg
        self.lr_scheduler = lr_scheduler

        # Learning rate decay scheduler parameters
        self.reduce_ratio = 0.5
        self.decay_steps = 20
        self.init_lr = lr

        # 1Cycle learning rate parameters
        if self.lr_scheduler == '1cycle':
            self.lr = lr / 10

    def step(self, layer, dw, db):
        """
        Optimizer step which updated the given weights
        :param layer: layer class that holds its W, b, vw and vb
        :param dw: derivative with respect to W
        :param db: derivative with respect to b
        :return: None
        """
        # classical momentum
        new_velocity_w = (self.momentum * layer.vw) - (self.lr * dw + self.lr * self.l2_reg * dw)
        new_velocity_b = (self.momentum * layer.vb) - (self.lr * db.reshape(-1, 1))

        new_W = layer.W + new_velocity_w
        new_b = layer.b + new_velocity_b
        layer.update_w(new_W=new_W, new_b=new_b, new_vw=new_velocity_w, new_vb=new_velocity_b)

    def init_V(self, v):
        """
        The vector v that is transferred throughout the network in the back propagation step
        :return: None
        """
        self.V = copy.deepcopy(v)

    def update_V(self, new_v):
        """
        Update v with the last derivative from the backward stage
        :param new_v: new V from back-prop derivative with respect to the data of certain layer i
        :return: None
        """
        self.V = new_v  # * self.V  # Append to the beginning of the list for the multiplication later

    def lr_step_decay(self, epoch):
        """
        Scheduler to reduce learning rate by reduce_ratio every decay_step epochs.
        :param epoch: current epoch
        :return: updated learning rate
        """
        new_lr = self.init_lr * math.pow(self.reduce_ratio, math.floor((1 + epoch) / self.decay_steps))
        return new_lr

    def one_cycle_policy(self, epoch, n_epochs):
        """
        One cycle policy introduced by L. smith that creates a cyclic learning rate.
        We start with a lr which is 10 times smaller than the initial lr. Then, it increases linearly until it reaches
        the initial learning rate at total_epochs/2. Then, it start decreasing again and in the few last epochs the
        reduce is becomes sharper.
        :param epoch: current epoch number
        :param n_epochs: total number of epochs
        :return: updated learning rate
        """
        lr_switch_point = int(n_epochs * 0.9 / 2)
        lr_step = 0.9 * self.init_lr / lr_switch_point

        if 0 <= epoch < lr_switch_point:
            new_lr = self.lr + lr_step
        elif lr_switch_point <= epoch < lr_switch_point * 2:
            new_lr = self.lr - lr_step
        else:
            # epoch >= lr_switch_point * 2:
            additional_decay = 100/(n_epochs - lr_switch_point * 2)
            new_lr = self.init_lr / (additional_decay * (epoch - lr_switch_point * 2 + 1))
        return new_lr

    def update_lr(self, curr_epoch, n_epochs):
        """
        Update learning rate based on the selected scheduling method.
        :param curr_epoch: current epoch number
        :param n_epochs: total number of epochs
        :return: updated learning rate
        """
        if self.lr_scheduler == '1cycle':
            self.lr = self.one_cycle_policy(curr_epoch, n_epochs)
            print(f'Updated learning rate using 1cycle policy: {self.lr}')
        elif self.lr_scheduler == 'step_decay':
            self.lr = self.lr_step_decay(curr_epoch)
            print(f'Updated learning rate using step decay policy: {self.lr}')
        return self.lr
