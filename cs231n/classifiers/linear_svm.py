import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
  """
 

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:,j] = dW[:,j] + np.transpose(X)[:,i]
        dW[:,y[i]]-= np.transpose(X)[:,i]
        loss += margin


  loss /= num_train
  dW /= num_train
  dW += reg * W

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
 
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  	
  scores = X.dot(W)
  scores+=1
  scores=np.transpose(np.transpose(scores)-y)
  scores[scores<0]=0
  scores[np.arange(0, num_train),y]=0
  loss=np.sum(scores)/num_train
  loss += reg * np.sum(W * W)


  '''W=W.T
  X=X.T
  scores = np.dot(W, X) # also known as f(x_i, W)

  correct_scores = np.ones(scores.shape) * scores[y, np.arange(0, scores.shape[1])]
  deltas = np.ones(scores.shape)
  L = scores - correct_scores + deltas

  L[L < 0] = 0
  L[y, np.arange(0, scores.shape[1])] = 0 # Don't count y_i
  loss = np.sum(L)

  # Average over number of training examples
  num_train = X.shape[1]
  loss /= num_train

  # Add regularization
  loss += reg * np.sum(W * W)'''
  
  
  scores = X.dot(W)
  scores+=1
  scores=np.transpose(np.transpose(scores)-y)
  scores[scores<0]=0
  scores[scores>0]=1
  scores[np.arange(0, num_train),y]=0
  scores[np.arange(0, num_train),y]=-1*np.sum(scores,axis=1)
  dW=np.transpose(X).dot(scores)
  dW /= num_train
  dW += reg * W


  return loss, dW
