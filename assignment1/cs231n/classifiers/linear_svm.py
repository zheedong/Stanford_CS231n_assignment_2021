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
    dW = np.zeros(W.shape)  # initialize the gradient as zero   차원 : (3073,10)

    
    # compute the loss and the gradient
    num_classes = W.shape[1]        # 10
    num_train = X.shape[0]          # 3073 (because of bias trick)
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:       # j = y[i]는 패스
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:               # margin이 0보다 크다면?
                loss += margin           # loss에 margin을 더해준다
                # ADDED, shape (3073) j != y[i]인 row에 대해서는 x_i가 더해진다.
                dW[:,j] += X[i,:]
                dW[:,y[i]] -= X[i,:]    

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train         # ADDED, 평균값이지?

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W       # ADDED, R(W) = sum_k sum_l W_k,l ^ 2. 편미분하면 2 * W가 된다.


    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    pass

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

    scores = np.matmul(X, W)
    correct_class_score = scores[range(X.shape[0]), y]

    delta = np.ones(scores.shape)
    delta[range(X.shape[0]), y] = 0             # delta는 j = y[i] 인 경우 0이다. 어라?
    margin = scores - np.reshape(correct_class_score, (correct_class_score.shape[0], 1)) + delta       # j = y[i] 인 경우, scores[j] = scores[y[i]] 라서 둘의 차는 0이 된다. => delta만 0으로 설정해 주면 된다!!!
    margin = np.maximum(0, margin)    

    loss += np.mean(np.sum(margin, axis = 1))         # 평균 구해주기 - numpy vector 연산 활용

    loss += reg * np.sum(np.square(W))                 # regularization
    pass

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
    margin_except_itself = np.zeros(margin.shape)           # margin에서 1(margin > 0) 를 적용하자
    margin_except_itself[margin > 0] = 1                    # margin이 0보다 크면 1, 0보다 작으면 무조건 0 (X는 마지막에 곱해주기로 - j = y[i]던 아니던 X는 곱해줘야 한다)
    margin_except_itself[range(X.shape[0]), y] = -np.sum(margin_except_itself, axis = 1)    # X.shape[0] = 0:500, j = y[i] 인 경우를 처리. j = y[i]인 경우에도 1로 되어 있을 것. 하지만 -(sum(1(margin > 0)))이 되어야 한다.

    dW = np.matmul(X.T, margin_except_itself) / X.shape[0]      # X를 한번에 곱해주고, N으로 나눠준다

    dW += 2*reg*W           # regularzation의 편미분도 더해준다

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
