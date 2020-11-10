"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    #calculating the loss
    Z = np.dot(X,W)
    
    row_sum = 0
    max_val = 0

    col_len = len(Z[:,0])
    row_len = len(Z[0,:])

    y_hat = np.zeros_like(Z)

    label_matrix = np.eye(row_len)[y]

    #create y_hat matrix
    for j in range (col_len):
        max_val = np.max(Z[j,:])
        for k in range (row_len):
            row_sum += np.exp(Z[j,k] - max_val)   
        for i in range (row_len):
            y_hat[j,i] = (np.exp(Z[j,i] - max_val))/row_sum
        row_sum = 0
        max_val = 0
    
    #compute the loss function
    for j in range (col_len):
        for k in range (row_len):
            loss += - label_matrix[j,k] * np.log(y_hat[j,k])
    
    loss = loss/col_len + reg*np.sum(np.square(W))  
    
    #calculating the gradient   
 
    dW = X.T @ (y_hat - label_matrix) /col_len + reg* 2*W
    

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    #print("third x")
    #print(X)
    #calculating the loss
    Z = np.dot(X,W)
    

    col_len = len(Z[:,0])
    row_len = len(Z[0,:])
    
    
    
    #pablo = np.max(Z, axis=1)
    Z -= np.reshape(np.max(Z, axis=1),[col_len,1])
    
    classElement = -Z[np.arange(col_len), y]
    row_sumV = np.sum(np.exp(Z), axis=1)
    
    term2 = np.log(row_sumV)
    loss = classElement + term2
    loss = np.sum(loss)
    loss /= col_len 
    loss += 0.5 * reg * np.sum(np.square(W))

    
    coef = np.exp(Z) / np.reshape(row_sumV,[col_len,1])
    coef[np.arange(col_len),y] -= 1
    dW = X.T.dot(coef)
    dW /= col_len
    dW += reg*2*np.sum(W)
    #loss = (classElement - np.log(row_sumV))/col_len + reg*np.sum(np.square(W))   
    #print(loss)
    #calculating the gradient   
 
    #dW = X.T @ (y_hat - label_matrix) /col_len + reg* 2*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val, lr, rg):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best = [0,0]
    best_softmax = None
    all_classifiers = []
    learning_rates = [2.580e-7, 2.62e-7]
    regularization_strengths = [2.4e4, 2.8e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    #lr 2.592000e-07 reg 2.440000e+04 train accuracy: 0.354479 val accuracy: 0.353000
    """
    n = [10, 10]
    step_reg = (regularization_strengths[1] - regularization_strengths[0])/n[1]
    step_learn = (learning_rates[1]-learning_rates[0])/n[0]
    
    learning_rates = np.arange(learning_rates[0],learning_rates[1], step_learn)
    regularization_strengths = np.arange(regularization_strengths[0], regularization_strengths[1], step_reg)
    
    perc_counter = 0
    print("0.0 %")
    for i in range(len(learning_rates)):
        for j in range(len(regularization_strengths)):
            softmax = SoftmaxClassifier()
            
            loss_history = softmax.train(X_train, y_train, learning_rate=learning_rates[i], reg=regularization_strengths[j], num_iters=1000, verbose=False)
            y_trainPred = softmax.predict(X_train)
            y_valPred = softmax.predict(X_val)
            training_accuracy = np.mean(y_train == y_trainPred)
            validation_accuracy = np.mean(y_val == y_valPred)

            results[learning_rates[i],regularization_strengths[j]]=training_accuracy,validation_accuracy

            all_classifiers.append(softmax)

            if validation_accuracy > best_val:
                best_val = validation_accuracy
                best_softmax = softmax
                best[0]=learning_rates[i]
                best[1]=regularization_strengths[j]
                
            perc_counter += 1
            print(100*perc_counter/(len(learning_rates)*len(regularization_strengths)), "%")
    """
    
    softmax = SoftmaxClassifier()
            
    loss_history = softmax.train(X_train, y_train, learning_rate=lr, reg=rg, num_iters=1000, verbose=False)
    y_trainPred = softmax.predict(X_train)
    y_valPred = softmax.predict(X_val)
    training_accuracy = np.mean(y_train == y_trainPred)
    validation_accuracy = np.mean(y_val == y_valPred)

    #results[learning_rates[i],regularization_strengths[j]]=training_accuracy,validation_accuracy

    best_val = validation_accuracy
    best_softmax = softmax
    print("lr = ", lr, " rg = ",rg)
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
