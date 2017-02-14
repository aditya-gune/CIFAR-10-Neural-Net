"""
INSERT YOUR NAME HERE
"""


from __future__ import division
from __future__ import print_function

import sys
import math
import cPickle
import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        print("F")
    # DEFINE __init function

    def forward(self, x):
        print("F")
    # DEFINE forward function

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        print("F")
    # DEFINE backward function
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def __init__(self, x):
        self.x = x
        
    def forward(self, x):
        x = np.maximum(0,x)
        return x
    # DEFINE forward function

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        print("F")
    # DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def forward(self, x, y):
        return self.y*np.log(self.x)+(1-self.y)*np.log(1-self.x)
        # DEFINE forward function
    def getSoftmax(self, x):
        return (1/(1+np.exp(x)))
    
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
        print("F")
        # DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
        self.input_dims = input_dims
        self.hidden_units = hidden_units
    # INSERT CODE for initializing the network

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):
        print("F")
    # INSERT CODE for training the network

    def evaluate(self, x, y):
        print("F")
    # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

if __name__ == '__main__':

    #data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    #train_x = data['train_data']
    #train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    hidden_units = 3
    mlp = MLP(input_dims, hidden_units)
    
    
    train_x = [[1,1,2],[2,1,1],[1,1,1],[2,2,2]]
    train_x = np.array(train_x)
    train_y=[[.001],[.999],[.999],[.001]]
    train_y = np.array(train_y)
    num_examples, input_dims = train_x.shape
    weights_h = [[-0.1, 1,0.1],[1,0.1,-1],[0,-0.1,1]]
    w_2 = [[-0.6],[0.9],[-0.3]] #[[-0.6, 0.4],[0.9,0.1],[-0.3, -0.8]]
    w_2 = np.array(w_2)
    w = np.array(weights_h)
    
    b = [0.01, 0.5, -0.11]

    b=np.array(b)
    print("linear transform input layer -> hidden layer")
    g = numpy.dot(train_x, w)+b
    print(g)
    relu = ReLU(g)
    print("output of hidden layer after relu")
    hidden_out = relu.forward(g)
    print(hidden_out)
    print("linear transform hidden layer -> output layer")
    out = numpy.dot(hidden_out, w_2)+0.01
    print(out)
    sce = SigmoidCrossEntropy(out, train_y)
    print("after softmax")
    #softmax = sce.getSoftmax(out)
    softmax = sce.getSoftmax(-out)
    print(softmax)
    print("Get cross entropy loss")
    loss = train_y*np.log(softmax)+((1-train_y)*np.log(1-softmax))
    loss = -loss
    print(loss)
    avg_loss = np.mean(loss)
    print("average loss")
    print(avg_loss)
#    for epoch in xrange(num_epochs):
#        
#    # INSERT YOUR CODE FOR EACH EPOCH HERE
#
#        for b in xrange(num_batches):
#            total_loss = 0.0
#            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
#            # MAKE SURE TO UPDATE total_loss
#            print(
#                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
#                    epoch + 1,
#                    b + 1,
#                    total_loss,
#                ),
#                end='',
#            )
#            sys.stdout.flush()
#        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
#        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
#        print()
#        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
#            train_loss,
#            100. * train_accuracy,
#        ))
#        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
#            test_loss,
#            100. * test_accuracy,
#        ))
