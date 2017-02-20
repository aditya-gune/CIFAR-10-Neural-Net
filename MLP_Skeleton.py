"""
Aditya Gune
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
        exps = np.exp(x)
        prob = exps /np.sum(exps, axis=1, keepdims=True)
        return prob
    
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
def normalize(z, avg, stdev):
    outctr = 1
    while outctr > 0:
        outctr = len(z[np.where(abs(z - avg) > stdev)])
        print("outctr = ", outctr, "\nnormalizing...")
        z[np.where(abs(z - avg) > stdev)] = avg
        outctr = len(z[np.where(abs(z - avg) > stdev)])
        print("outctr = ", outctr)

def debugprint(softmax):
    print ("max = ",np.max(softmax), "min =", np.min(softmax))
    print ("mean = ", np.mean(softmax), "std = ", np.std(softmax))
    
       
if __name__ == '__main__':

    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']
    
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    hidden_units = 3
    num_examples, input_dims = (int(train_x.shape[0]), int(train_x.shape[1]))
    mlp = MLP(input_dims, hidden_units)
    alpha = 0.001 #learning rate
    r_lambda = 0.001 #reg lambda
    
    
    weights_h = np.random.random((input_dims, 1)) * 2.0/input_dims
    w = np.array(weights_h)
    w_2 = np.random.random((input_dims, 1)) * 2.0/input_dims
    w_2 = np.array(w_2)

    
    
    b = np.random.random((1, train_x.shape[1]))
    b_2 = np.random.random((1, train_x.shape[1]))
    b=np.array(b)
    b_2 = np.array(b_2)
    #Apply linear transform, send to hidden layer:
    # input layer -> hidden layer
    g = np.dot(train_x, w)+b

    
    #Apply ReLU Activation function
    relu = ReLU(g)
    z = relu.forward(g)

    #normalize data 
    stdev = np.std(z)
    avg = np.mean(z)
    
#    print ("max = ",np.max(z), "min =", np.min(z))
#    print ("mean = ", avg, "std = ", stdev)
#    
    #normalize(z, avg, stdev)
    
    #Apply linear transform to hidden layer
    out = np.dot(z, w_2)+b_2
    #print(out)
    
    sce = SigmoidCrossEntropy(out, train_y)
    softmax = sce.getSoftmax(-out)
    #softmax = (np.exp(out)#getSoftmax(-out)
#    print("after softmax")

    d_3 = softmax
    #d_3[range(num_examples), train_y] -= 1
    d_wt_2 = np.dot(z.T, d_3)
    normalize(d_wt_2, np.mean(d_wt_2), np.std(d_wt_2))
    
    
    d_bias_2 = np.sum(d_3, axis=0, keepdims=True)
    d_2 = np.dot(w_2.T, d_3.T)#* (1 - np.power(z, 2))
    
    d_wt_1 = np.dot(train_x.T, d_2.T)
    d_bias_1 = np.sum(d_3, axis=0)
    
    
    #debugprint(d_wt_1)
    
    #Gradient Descent
    weights_h += -alpha * d_wt_1
    w_2 = w_2 + (-alpha * d_wt_2[1])
    b = b+ (-alpha * d_bias_1)
    b_2 += -alpha * d_bias_2
    
    loss = train_y*np.log(softmax)+((1-train_y)*np.log(1-softmax))
    loss = -loss
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
