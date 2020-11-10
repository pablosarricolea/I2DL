"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
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
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        
        H1 = np.dot(X,W1)+b1
        Z1 = np.maximum(0,H1)
        scores = np.dot(Z1,W2) + b2  #scores  = H2
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################

        #apply the softmax to the H2 matrix
        #Z2 = activationfunction(H2)
        H2 = np.copy(scores)
        H2 -= np.reshape(np.max(H2, axis=1),[N,1])
        Z2  = np.exp(H2)
        softmax_sum = np.sum(Z2, axis=1)
        
        #reshape as a column vector 
        # | a11 a12 a13 |     | b1 |       | a11/b1 a12/b1 a13/b1 |
        # | a21 a22 a23 |  :  | b2 |  =    | a21/b2 a22/b2 a23/b2 |
        # | a31 a32 a33 |     | b3 |       | a31/b3 a32/b3 a33/b3 |
        
        Z2 /= softmax_sum.reshape(N,1)
        
        #selecting only the true labels and sum all the values of the matrix
        loss = - np.sum(np.log(Z2[np.arange(N), y]))
        
        #normalize
        loss /= N
        #add the regulation value
        loss += 0.5 * reg * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
        
        
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        
        #derive dL/dZ2
        dZ2 = np.copy(Z2)
        dZ2[np.arange(N), y] -= 1
        
        
        dH1 = np.dot(dZ2, W2.T) * (H1 > 0)
        dZ1 = np.dot(dZ2, W2.T) 
        
        # compute gradient for parameters
        grads['W2'] = np.dot(Z1.T, dZ2) / N      
        grads['b2'] = np.sum(dZ2, axis=0) / N      
        grads['W1'] = np.dot(X.T, dH1) / N        
        grads['b1'] = np.sum(dH1, axis=0) / N       

        # add reg term
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################

            index = np.random.choice(X.shape[0], batch_size)
            X_batch = X[index]
            y_batch = y[index]

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            
            
            self.params['W1']  = self.params['W1'] - learning_rate  * grads['W1']
            self.params['b1']  = self.params['b1'] - learning_rate  * grads['b1']
            self.params['W2']  = self.params['W2'] - learning_rate  * grads['W2']
            self.params['b2']  = self.params['b2'] - learning_rate  * grads['b2']
             

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

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
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None
        N, _ = X.shape
        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################

        H1 = np.dot(X,self.params['W1'])+self.params['b1']
        Z1 = np.maximum(0,H1)
        H2 = np.dot(Z1,self.params['W2']) + self.params['b2']  
        H2 -= np.reshape(np.max(H2, axis=1),[N,1])
        Z2 = np.exp(H2)
        softmax_sum = np.sum(Z2, axis=1)
        Z2 /= softmax_sum.reshape(N,1)
        
        y_pred = np.argmax(Z2, axis=1)

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    input_size = 32 * 32 * 3
    hidden_size = 50
    num_classes = 10
    
    best_acc = -1
    best_net = None
    results ={}
    best_values = [0,0,0] #[lr,rg,bs]
    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above in the Jupyther Notebook; these visualizations   #
    # will have significant qualitative differences from the ones we saw for   #
    # the poorly tuned network.                                                #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    lr = np.linspace(1.2e-3,1.4e-3,3)
    rg = np.linspace(0.05,0.1,2)
    bs = [350,400]
    count=0
    
    for i in lr:
        for j in rg:
            for k in bs:
                count += 1
                net = TwoLayerNet(input_size, hidden_size, num_classes)
                stats = net.train(X_train, y_train, X_val, y_val, num_iters=1000, batch_size=k, learning_rate=i, learning_rate_decay=0.95, reg=j, verbose=False)
                
                val_acc = (net.predict(X_val) == y_val).mean()
                train_acc = (net.predict(X_train) == y_train).mean()
                
                results[i, j, k]=val_acc,train_acc

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_net=net
                    best_values[0]=i
                    best_values[1]=j
                    best_values[2]=k
                print(100*count/(len(lr)*len(rg)*len(bs)), "%")
    # Predict on the validation set
          
    print("lr = ", best_values[0], "rs = ", best_values[1], "bs = ", best_values[2], "val acc = ",best_acc)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
