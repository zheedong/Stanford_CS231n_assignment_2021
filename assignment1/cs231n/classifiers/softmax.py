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

    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)                # For numeric instability

        scores_exp = np.sum(np.exp(scores))     # sum_j e^{f_j}
        correct_exp = np.exp(scores[y[i]])      # e^{f_{y_i}}

        loss -= np.log(correct_exp / scores_exp)        # L_i = - log (sum_j e^{f_j} / e^{f_{y_i}})

        for j in range(num_classes):
            if j == y[i]:
                continue
            dW[:,j] += np.exp(scores[j]) / scores_exp *  X[i]               # For j != y_i, dW = e^{f_{j}} / sum_j e^{f_j} * x_i
        dW[:,y[i]] -= (scores_exp - correct_exp) / scores_exp * X[i]        # For j == y_i, dW = - (sum_j e^{f_j} - e^{f_{y_i}}) / sum_j e^{f_{y_i}} * x_i

    loss /= num_train               # divide by N
    dW /= num_train                 # divide by N

    loss += reg * np.sum(W*W)       # Add Regularization R(W) = sum W^2
    dW += 2*reg*W                   # Add Regularization gradient

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
    
    # f(x_i, W) = Wx_i
    num_train = X.shape[0]

    scores = np.matmul(X,W)         # f (scores) = f(X, W) = WX
    scores -= np.max(scores)        # For numeric instability

    sum_exp_scores = np.exp(scores).sum(axis = 1, keepdims = True)
    softmax_matrix = np.exp(scores) / sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]))     # 여기가 이해 안 됨!

    softmax_matrix[np.arange(num_train), y] -= 1        # Correct Case인 경우는 다르다
    dW = np.matmul(X.T, softmax_matrix)

    loss /= num_train       # 평균 구하기
    dW /= num_train

    loss += reg * np.sum(W * W)     # Regularization
    dW += reg * 2 * W               # 편미분해서 들어감

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
