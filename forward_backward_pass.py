def forward_pass(X, net, softmax):
    """
    This functin applies the forward pass of the network, it starts with the input X, and pass it the the first
    layer. Then, each layer calculates its output and passes it as input to the next layer. At the end we have
    the last layer followed by softmax function.
    :param X: input to the network
    :param net: network layers
    :param softmax: softmax function
    :return: The output of the layer+softmax
    """
    out = X
    for layer in net:
        out = layer.forward(out)
    softmax_out = softmax.forward(out)
    return softmax_out


def backward_pass(Yt, net, optim, sm_out, softmax, mode='train'):
    """
    This function computed the Backward pass and updates weights, biases, etc.
    :param Yt: Batch labels
    :param net: network
    :param optim: optimizer
    :param sm_out: softmax output
    :param softmax: softmax function
    :param mode: mode in ('train', 'grad_ver'). If mode == 'grad_ver', we save the gradient of each layer
    for the verification process.
    :return: None
    """
    sm_dw, sm_db = softmax.backward_w(sm_out, Yt)
    sm_dx = softmax.backward_x(sm_out, Yt)
    optim.step(softmax, sm_dw, sm_db)
    optim.init_V(sm_dx)

    if mode == 'grad_ver':
        grads = [(sm_dw, sm_db)]
    else:
        grads = None

    for layer_b in net[::-1]:
        # Calc derivatives for weights and biases using the "Jacobian^T times vector
        JWtV, JbtV = layer_b.backward_w(optim.V)
        JXtV = layer_b.backward_x(optim.V)

        # Save JXtV for next calculations
        optim.update_V(JXtV)

        # Make optimization step to update layer weights
        optim.step(layer_b, JWtV, JbtV)

        # Save gradients for gradient verification
        if mode == 'grad_ver':
            grads.append((JWtV, JbtV))

    return grads
