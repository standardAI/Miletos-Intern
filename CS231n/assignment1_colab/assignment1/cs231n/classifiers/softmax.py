from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # compute the loss and the gradient
    C = W.shape[1]
    N = X.shape[0]
    
    for i in range(N):
        scores = X[i].dot(W)                                                    
        correct_class_score = scores[y[i]]
        denominator = np.exp(scores).sum()
        loss += -np.log(np.exp(correct_class_score)/denominator)
        
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        for j in range(C):
            S_j = np.exp(scores[j])/denominator
            
            if j == y[i]:
                dW[:, j] += (S_j - 1)*X[i]
            else:
                dW[:, j] += (S_j - 0)*X[i]
    
    def softmax_stable(x):
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
            

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N
    dW /= N

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    scores = X.dot(W)  # (N, C)
    correct_class_scores = scores[range(N), y]  # (N,)
    e = np.exp(scores)
    denominators = e.sum(axis=1)  # (N,)
    #a = np.where(denominators == 0)
    #if a:
    #    print(0)
    loss = -np.log(np.exp(correct_class_scores)/denominators)  # (N, 1)
    loss = loss.sum()/N
    loss += reg * np.sum(W * W)
    
    softmaxes = e / denominators.reshape(-1, 1)   # (N, C)
    softmaxes[range(N), y] -= 1
    dW = X.T.dot(softmaxes)  # (D, N).dot((N, C)) -> (D, C)
    dW /= N
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
