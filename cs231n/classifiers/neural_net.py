from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
 

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
   
    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of loss and grads
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    z1=X.dot(W1)+b1
    a1=np.maximum(0,z1)   #Relu activation fn.(hidden layer), z1(N , hidden layer size) is the input to hidden layer
    z2=a1.dot(W2)+b2
    
    
    '''e=np.exp(z2)
    sumexp=np.sum(e,axis=1)             
    scores=e/np.reshape(sumexp,(-1,1))    #Softmax activation fn(output layer), z2 (N, output layer size(no.of classes)) is the                                           #input to output layer
                      
                      
                       '''
    
    
    if y is None:
      return z2    #Z2 contains the scores

    # Compute the loss
    loss = None
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

    # average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(N), y])
    data_loss = np.sum(corect_logprobs) / N
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    
    '''
    softmax_loss = None
    arr=scores[:,y]
    softmax_loss=np.sum(-1*np.log(arr))/N          #softmax loss
    softmax_loss+=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))'''
  
    # Backward pass: compute gradients
    grads = {}
    dscores = z2
    dscores[range(N),y] -= 1
    dscores /= N

    # W2 and b2
    grads['W2'] = np.dot(a1.T, dscores)
    grads['b2'] = np.sum(dscores, axis=0)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[a1 <= 0] = 0
    # finally into W,b
    grads['W1'] = np.dot(X.T, dhidden)
    grads['b1'] = np.sum(dhidden, axis=0)

    # add regularization gradient contribution
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        
    
   
    # Train this neural network using stochastic gradient descent.
    # -learning_rate_decay: Scalar giving factor used to decay the learning rate
    # after each epoch.
    #- reg: Scalar giving regularization strength.
    #- num_iters: Number of steps to take when optimizing.
    #- batch_size: Number of training examples to use per step.
    #- verbose: boolean; if true print progress during optimization.

   
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)     #no. of steps taken per epoch

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      a=np.random.randint(num_train,size=batch_size)
      X_batch = X[a,:]
      y_batch = y[a]

      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
      self.params['W1']+= -learning_rate*grads['W1']
      self.params['b1']+= -learning_rate*grads['b1']
      self.params['W2']+= -learning_rate*grads['W2']
      self.params['b2']+= -learning_rate*grads['b2']
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """ 
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    scores = None
    z1=X.dot(W1)+b1
    outrel=np.maximum(0,z1)
    z2=outrel.dot(W2)+b2
    e=np.exp(z2)
    sumexp=np.sum(e,axis=1)
    scores=e/np.reshape(sumexp,(-1,1))	
    y_pred = np.argmax(scores,axis=1)

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


