from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

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
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    # The network is architected to have 2 layers, and each layer has the form of:
    # output = input * weight + bias

    # For layer 1 the "input" is the raw training data, and the output is a vector of
    # activated scores.

    # We take the dot product here because X is a matrix of NxD dimension and the weight 
    # matrix is a matrix of DxH dimension, resulting in a matrix of NxH dimension
    layer1Output = X.dot(W1) + b1

    # Now we must use the ReLU to cull any negative values befor sending the layer 1
    # values off to layer 2.
    # Note this NP function operates on the entire vector of layer1Output
    layer2Input = np.maximum(0, layer1Output)

    # For layer 2 the "input" is the output from layer 1, and the output
    # is the final scores for the classification.

    # The dot product here works because layer2Input is a matrix of NxH dimension and the 
    # weight matrix is a matrix of HxC dimension, resulting in a vector of NxC dimension
    layer2Output = layer2Input.dot(W2) + b2

    scores = layer2Output
  
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    # First let us compute the softmax loss
    # Softmax loss has a formula where the loss for any individual data point is 
    # L_i = -log(e^data_i/sumforallData(e^data_i))
    # To get loss for an entire set then, we sum all softmaxes, take the log after
    # and divide by the number of data points to get the average

    # Note we subtract the max off of the data set to ensure numeric stablity
    # See slide 34 of the w3-2_ML_logistic_regression deck from class

    exps = np.exp(scores - np.max(scores))
    # The keep axis and dimiension ensures we are summing across the right values in scores
    softMax = exps / np.sum(exps, axis=1, keepdims=True)

    # This arrange function just allows us to access the correct indices of softMax
    # in an efficient way
    loss = np.sum(-np.log(softMax[np.arange(N), y]))

    # To take the average, divide by the number of data points
    loss /= N

    # Now to add the L2 regulation
    # L2 regulation is calculated using the sum of the square of all the individual
    # elements of the weight matrix, and python does nice matrix squaring by just
    # multiplying the two matrices together!

    # Since we have two weight matrices, we can just calculate the L2 regulation for
    # both and sum them together
    W1Reg = np.sum(W1 * W1)
    W2Reg = np.sum(W2 * W2)

    totalReg = W1Reg + W2Reg

    # Now we just scale this by the given regulation hyper parameter 'reg' and add
    # it to the loss scalar.

    loss += reg * totalReg
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    # We must work backwards from the softmax loss all the way back to the input, 
    # taking partial derivatives along the way. 

    ######################################
    # THE DERIVATIVE OF SOFTMAX LOSS WRT SCORES
    ######################################

    # This derivative was found to be computed according to this resource 
    # https://deepnotes.io/softmax-crossentropy
    # The code snippet was also inspired from that resource

    # start by copying it over
    diLossdiScore =  softMax
    diLossdiScore[np.arange(N) ,y] -= 1
    diLossdiScore /= N

    ######################################
    # THE DERIVATIVE OF LOSS WRT W2
    ######################################

    # We note that the scores are calculated as Score = W2*layer2Input + b2
    # The partial derrivative is then: 
    # d Loss/d Score * d Score/d W2

    # We find d Score/d W2 as just the layer2Input vector! To calculate the 
    # sum of all products of d Loss/d Score * d Score/d W2 across all elements, 
    # we can transpose layer2Input and multiply it with diLossdiScore
    diLossdiW2 = layer2Input.T.dot(diLossdiScore)


    ######################################
    # THE DERIVATIVE OF LOSS WRT B2
    ######################################

    # We again note that the scores are calculated as Score = W2*layer2Input + b2
    # The partial derrivative is then: 
    # d Loss/d Score * d Score/d B2

    # We find d Score/d B2 as just 1. To calculate the 
    # sum of all products of d Loss/d Score * d Score/d W2 across all elements
    # we actually just end up summing d Loss/d Score

    diLossdiB2 = np.sum(diLossdiScore, axis=0)

    ######################################
    # THE DERIVATIVE OF LOSS WRT W1
    ######################################

    # We again note that the scores are calculated as Score = W2*layer2Input + b2
    # We also note that layer2Input is calculated as layer2Input = max(0, W1*inputData + B1)
    # The partial derivative then is 
    # d Loss/d Score * d Score/d layer2Input * d layer2Input/d W1

    # We find d Loss/d Score * d Score/d layer2Input as just d Loss/d Score multiplied by W2

    # We find d layer2Input/d W1 as inputData, while also culling the negative values
    # because we did the ReLU between the two layers. Again we use transpose and multiply
    # do do an efficient sum.

    diLossdiW1 = softMax.dot(W2.T)
    positiveCull = diLossdiW1 * (layer1Output > 0)
    diLossdiW1 = X.T.dot(positiveCull)

    ######################################
    # THE DERIVATIVE OF LOSS WRT B2
    ######################################

    # Following the same observations as above, we find the partial derivative breakdown.
    # The partial derivative then is 
    # d Loss/d Score * d Score/d layer2Input * d layer2Input/d B1

    # We find d Loss/d Score * d Score/d layer2Input as just d Loss/d Score multiplied by W2

    # We find d layer2Input/d W1 as 1, and end up finding the whole partial derivative as 
    # just the sum of positive culled values of d Loss/d Score * d Score/d layer2Input using 
    # transpost and multiply. 

    diLossdiB1 = positiveCull.sum(axis=0)

    ######################################
    # ACCOUNTING FOR REGULATION
    ######################################

    # Because we added regulations to the losses, and they are functions of W1 and W2 respectively,
    # we must add these to diLossdiW1 and diLossdiW2. The derivates are trivial, and found to just
    # be solved using power rule. 

    diLossdiW1 += reg * 2 * W1
    diLossdiW2 += reg * 2 * W2

    # Now we just append to the dictionary and return!
    grads = {'W1':diLossdiW1, 'b1':diLossdiB1, 'W2':diLossdiW2, 'b2':diLossdiB2}
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

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
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


