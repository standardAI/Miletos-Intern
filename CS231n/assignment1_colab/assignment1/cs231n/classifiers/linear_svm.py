from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    N = X.shape[0]
    C = W.shape[1]
    loss = 0.0
    for i in range(N):
        scores = X[i].dot(W)                                                    #scores = X.dot(W)  (N,D).dot(D,C) -> (N,C)
        score_correct = scores[y[i]]                                            #correct_class_scores = scores[np.arange(len(scores)), y]
        for j in range(C):                                             
            if j == y[i]:
                continue

            margin = scores[j] - score_correct + 1  # note delta = 1            #margin = scores - correct_class_scores + 1  (1,C) - (1,)
            if margin > 0:                                                      #margin[y[i]] = 0;  margin[margin < 0] = 0;
                loss += margin                                                  #loss += np.sum(margin)
                dW[:, y[i]] -= X[i]                                             #dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]                                                #dW[:, margin > 0] += X[i]  (D,C) + (1,D)
                

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= N
    dW /= N

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

   
    
    
    scores = X.dot(W)  # (N,D).dot(D,C) -> (N,C)
    correct_class_scores = scores[np.arange(len(scores)), y]  # (N,1)
    margin = scores - correct_class_scores.reshape(-1,1) + 1  # (N,C) - (N,1) + (1,1) -> (N,C)
    margin[np.arange(len(margin)), y] = 0  # (N,C)
    margin[margin < 0] = 0  # (N,C)
    loss = np.sum(margin)  # (1,1)
    loss /= X.shape[0]
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    margin[margin > 0] = 1  # This is the gradient with respect to the row of W that doesn't correspond to the correct class
    margin[np.arange(len(margin)), y] = -np.sum(margin, axis=1)  # Counting the number of classes that didn't meet the desired margin
    dW = X.T.dot(margin)  # (D, N).dot(N, C) -> (D, C)
    dW /= X.shape[0]
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
