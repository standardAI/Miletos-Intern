from builtins import range
from this import d
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.reshape(x.shape[0], -1).dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout.dot(w.T).reshape(x.shape[0], *x.shape[1:])
    dw = dout.T.dot(x.reshape(x.shape[0], -1)).T
    db = np.ones((1, dout.shape[0])).dot(dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = np.zeros_like(cache), cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = 1*dout
    dx[x < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    loss = 0.0
    correct_class_scores = x[range(x.shape[0]), y]  # (N, 1) for each training example
    denominators = np.exp(x).sum(axis=1)  # (N, 1)
    loss = -np.log(np.exp(correct_class_scores)/denominators)  # (N, 1)
    loss = loss.sum()/x.shape[0]
    # loss += reg * np.sum(W * W)

    numerators = np.exp(x) / denominators.reshape(-1, 1)
    numerators[range(x.shape[0]), y] -= 1
    dx = numerators
    dx /= x.shape[0]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = np.mean(x, axis=0)  # x: (N, D) -> For each N
        sample_var = np.var(x, axis=0)
        x_batchnormalized = (x - sample_mean) / (np.sqrt(sample_var + eps))
        out = gamma * x_batchnormalized + beta
        cache = (x, x_batchnormalized, sample_mean, sample_var, gamma, beta, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_batchnormalized = (x - running_mean) / (np.sqrt(running_var) + eps)
        out = gamma * x_batchnormalized + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_batchnormalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    # derivative of gamma * x_batchnormalized + beta
    dxbatchnormalized = dout * gamma
    # derivative of (x - sample_mean) / (np.sqrt(sample_var + eps))
    dsample_var = np.sum(dxbatchnormalized * (x - sample_mean) * (-1/2) * (sample_var + eps)**(-3/2), axis=0)
    dsample_mean = np.sum(dxbatchnormalized * (-1/np.sqrt(sample_var + eps)), axis=0) + dsample_var * np.sum(-2*(x - sample_mean), axis=0)/N
    dx = dxbatchnormalized * 1 / np.sqrt(sample_var + eps) + dsample_var * 2 * (x - sample_mean) / N + dsample_mean / N
    dgamma = np.sum(dout * x_batchnormalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_batchnormalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    # derivative of gamma * x_batchnormalized + beta
    dxbatchnormalized = dout * gamma
    dx = -1 / sample_var + ((2 / N) * np.sum(x - sample_mean, axis=0) * (1/(2 * np.sqrt(sample_var + eps))) * ((sample_mean - x)/(sample_var**2)))
    dx = dx * dout

    dgamma = np.sum(dout * x_batchnormalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sample_mean = np.mean(x, axis=1)  # (N, D) -> (N, 1)
    sample_var = np.var(x, axis=1)
    #print(x.shape, sample_mean.reshape(-1, 1).shape)
    x_layernormalized = (x - sample_mean.reshape(x.shape[0], 1)) / (np.sqrt(sample_var + eps)).reshape(-1, 1)
    out = gamma * x_layernormalized + beta
    cache = (x, x_layernormalized, sample_mean, sample_var, gamma, beta, eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_layernormalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    # derivative of gamma * x_batchnormalized + beta
    dxbatchnormalized = dout * gamma
    # derivative of (x - sample_mean) / (np.sqrt(sample_var + eps))

    dsample_var = np.sum(dxbatchnormalized * (x - sample_mean.reshape(-1, 1)) * (-1/2) * ((sample_var + eps)**(-3/2)).reshape(-1, 1), axis=1)
    dsample_mean = np.sum(dxbatchnormalized * (-1/np.sqrt(sample_var + eps)).reshape(-1, 1), axis=1) + dsample_var * np.sum(-2*(x - sample_mean.reshape(-1, 1)), axis=1)/N
    dx = dxbatchnormalized * 1 / np.sqrt(sample_var + eps).reshape(-1, 1) + dsample_var.reshape(-1, 1) * 2 * (x - sample_mean.reshape(-1, 1)) / N + dsample_mean.reshape(-1, 1) / N
    dgamma = np.sum(dout * x_layernormalized, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = mask * x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modify the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_, W_))
    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))  # pad to 2. and 3. dimentions
    x_padded = np.pad(x, npad, "constant")

    for i in range(N):
        for j in range(F):
            for k in range(0, H_*stride, stride):
                for l in range(0, W_*stride, stride):
                    out[i, j, int(k/stride), int(l/stride)] = np.sum(x_padded[i, :, k:k+HH, l:l+WW] * w[j]) + b[j]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    H_ = 1 + (H + 2 * pad - HH) // stride
    W_ = 1 + (W + 2 * pad - WW) // stride
    npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))  # pad to 2. and 3. dimentions
    x_padded = np.pad(x, npad, "constant")

    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    for i in range(N):
        for j in range(F):
            for k in range(0, H_*stride, stride):
                for l in range(0, W_*stride, stride):
                    dx_padded[i, :, k:k+HH, l:l+WW] += w[j] * dout[i, j, int(k/stride), int(l/stride)]
                    dw[j] += x_padded[i, :, k:k+HH, l:l+WW] * dout[i, j, int(k/stride), int(l/stride)]
                    db[j] += dout[i, j, int(k/stride), int(l/stride)]

    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]  #To return orijinal input shape
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_, W_))
    for i in range(N):
        for j in range(C):
            for k in range(0, H_*stride, stride):
                for l in range(0, W_*stride, stride):
                    out[i, j, int(k/stride), int(l/stride)] = np.max(x[i, j, k:k+pool_height, l:l+pool_width])
                    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    H_ = 1 + (H - pool_height) // stride
    W_ = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(0, H_*stride, stride):
                for l in range(0, W_*stride, stride):
                    x_pool = x[i, j, k:k+pool_height, l:l+pool_width]
                    dx[i, j, k:k+pool_height, l:l+pool_width] += (x_pool == np.max(x_pool)) * dout[i, j, int(k/stride), int(l/stride)]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, C, H, W = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(C, dtype=x.dtype))

    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sample_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)  # (N, C, H, W) - > rather than (C,) ->> (1, C, 1, 1)
    sample_var = np.var(x, axis=(0, 2, 3), keepdims=True)
    x_spatialbatchnormalized = (x - sample_mean) / (np.sqrt(sample_var + eps))
    out = gamma.reshape(1, C, 1, 1) * x_spatialbatchnormalized + beta.reshape(1, C, 1, 1)
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    cache = (x, x_spatialbatchnormalized, sample_mean, sample_var, gamma, beta, bn_param)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    """
    N, C, H, W = dout.shape
    x, x_spatialbatchnormalized, sample_mean, sample_var, gamma, beta, bn_param, eps = cache
    eps = bn_param.get("eps", 1e-5)
    #print("gamma's shape: ", gamma.shape)
    #print("gamma0's value: ", gamma[0])
    gamma0, gamma1, gamma2 = gamma[0], gamma[1], gamma[2]
    beta0, beta1, beta2 = beta[0], beta[1], beta[2]
    sample_mean0, sample_mean1, sample_mean2 = sample_mean[0,0,0,0].reshape(1,1), sample_mean[0,1,0,0].reshape(1,1), sample_mean[0,2,0,0].reshape(1,1)
    sample_var0, sample_var1, sample_var2 = sample_var[0,0,0,0].reshape(1,1), sample_var[0,1,0,0].reshape(1,1), sample_var[0,2,0,0].reshape(1,1)
    x0 = x[:, 0, :, :].reshape(N, -1)
    x1 = x[:, 1, :, :].reshape(N, -1)
    x2 = x[:, 2, :, :].reshape(N, -1)
    x_spatialbatchnormalized0 = x_spatialbatchnormalized[:, 0, :, :].reshape(N, -1)
    x_spatialbatchnormalized1 = x_spatialbatchnormalized[:, 1, :, :].reshape(N, -1)
    x_spatialbatchnormalized2 = x_spatialbatchnormalized[:, 2, :, :].reshape(N, -1)
    #print("gamma0's shape: ", gamma0.shape)
    cache = x0, x_spatialbatchnormalized0, sample_mean0, sample_var0, gamma0, beta0, eps
    #print(dout[:, 0, :, :].reshape(N, -1).shape)
    dx0, dgamma0, dbeta0 = batchnorm_backward(dout[:, 0, :, :].reshape(N, -1), cache)
    cache = x1, x_spatialbatchnormalized1, sample_mean1, sample_var1, gamma1, beta1, eps
    dx1, dgamma1, dbeta1 = batchnorm_backward(dout[:, 1, :, :].reshape(N, -1), cache)
    cache = x2, x_spatialbatchnormalized2, sample_mean2, sample_var2, gamma2, beta2, eps
    dx2, dgamma2, dbeta2 = batchnorm_backward(dout[:, 2, :, :].reshape(N, -1), cache)
    #print(dgamma0.shape, dbeta0.shape)
    dx = np.concatenate((dx0.reshape(N, 1, H, W), dx1.reshape(N, 1, H, W), dx2.reshape(N, 1, H, W)), axis=1)
    dgamma = np.concatenate((dgamma0.sum().reshape(1,1), dgamma1.sum().reshape(1,1), dgamma2.sum().reshape(1,1)))
    dbeta = np.concatenate((dbeta0.sum().reshape(1,1), dbeta1.sum().reshape(1,1), dbeta2.sum().reshape(1,1)))
    """

    x, x_batchnormalized, sample_mean, sample_var, gamma, beta, bn_param = cache
    eps = bn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    # derivative of gamma * x_batchnormalized + beta
    dxbatchnormalized = dout * gamma.reshape(1, C, 1, 1)
    # derivative of (x - sample_mean) / (np.sqrt(sample_var + eps))
    dsample_var = np.sum(dxbatchnormalized * (x - sample_mean) * (-1/2) * (sample_var + eps)**(-3/2), axis=0)
    dsample_mean = np.sum(dxbatchnormalized * (-1/np.sqrt(sample_var + eps)), axis=0) + dsample_var * np.sum(-2*(x - sample_mean), axis=0)/N
    dx = dxbatchnormalized * 1 / np.sqrt(sample_var + eps) + dsample_var * 2 * (x - sample_mean) / N + dsample_mean / N
    dgamma = np.sum(dout * x_batchnormalized, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))

    """
    x, x_batchnormalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, C, H, W = x.shape
    # derivative of gamma * x_batchnormalized + beta
    dxspatialbatchnormalized = dout * gamma.reshape(1, C, 1, 1)
    # derivative of (x - sample_mean) / (np.sqrt(sample_var + eps))
    dsample_var = np.sum(dxspatialbatchnormalized * (x - sample_mean.reshape(1, C, 1, 1)) * (-1/2) * (sample_var.reshape(1, C, 1, 1) + eps)**(-3/2), axis=(0, 2, 3), keepdims=True)
    dsample_mean = np.sum(dxspatialbatchnormalized * (-1/np.sqrt(sample_var + eps)), axis=(0, 2, 3), keepdims=True) + dsample_var * np.sum(-2*(x - sample_mean), axis=(0, 2, 3), keepdims=True)/N
    dx = dxspatialbatchnormalized * 1 / np.sqrt(sample_var.reshape(1, C, 1, 1) + eps) + dsample_var.reshape(1, C, 1, 1) * 2 * (x - sample_mean.reshape(1, C, 1, 1)) / N + dsample_mean.reshape(1, C, 1, 1) / N
    dgamma = np.sum(dout * x_batchnormalized, axis=(0, 2, 3))
    dbeta = np.sum(dout, axis=(0, 2, 3))
    """

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
    - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)

    momentum = gn_param.get("momentum", 0.9)

    N, C, H, W = x.shape
    running_mean = gn_param.get("running_mean", np.zeros(C//G, dtype=x.dtype))
    running_var = gn_param.get("running_var", np.zeros(C//G, dtype=x.dtype))
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # TODO: Conver this to a for loop
    sample_mean1 = np.mean(x[:, :C//G, :, :], axis=(1, 2, 3), keepdims=True)
    sample_var1 = np.var(x[:, :C//G, :, :], axis=(1, 2, 3), keepdims=True)
    x_batchnormalized1 = (x[:, :C//G, :, :] - sample_mean1) / np.sqrt(sample_var1 + eps)
    out1 = x_batchnormalized1 * gamma[:, :C//G, :, :] + beta[:, :C//G, :, :]

    sample_mean2 = np.mean(x[:, C//G:, :, :], axis=(1, 2, 3), keepdims=True)
    sample_var2 = np.var(x[:, C//G:, :, :], axis=(1, 2, 3), keepdims=True)
    x_batchnormalized2 = (x[:, C//G:, :, :] - sample_mean2) / np.sqrt(sample_var2 + eps)
    out2 = x_batchnormalized2 * gamma[:, C//G:, :, :] + beta[:, C//G:, :, :]

    out = np.concatenate((out1, out2), axis=1)

    running_mean[0] = momentum * running_mean[0] + (1 - momentum) * sample_mean1
    running_var[0] = momentum * running_var[1] + (1 - momentum) * sample_var1
    gn_param["running_mean"] = running_mean
    gn_param["running_var"] = running_var
    cache1 = (x[:, :C//G, :, :], x_batchnormalized1, sample_mean1, sample_var1, gamma[:, :C//G, :, :], beta[:, :C//G, :, :], eps)
    cache2 = (x[:, C//G:, :, :], x_batchnormalized2, sample_mean2, sample_var2, gamma[:, C//G:, :, :], beta[:, C//G:, :, :], eps)
    cache = (cache1, cache2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
