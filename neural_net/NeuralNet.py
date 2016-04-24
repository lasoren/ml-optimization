"""
Created on Wed Feb 10 21:56:02 2016

@author: Ryan Lader, Emily MacLeod working from Lab4_Soln
"""

import numpy as np
import matplotlib.pyplot as plt

def construct_truth(y):
    v = []
    for i in range(len(y)):
        if y[i] == 0:
            v += [[1,0]]
        elif y[i] == 1:
            v += [[0,1]]
    return v

class NeuralNet:
    """
    This class implements a simple 3 layer neural network.
    """
    
    def __init__(self, input_dim, hidden_layer, hidden_dim, output_dim, epsilon):
        """
        Initializes the parameters of the neural network to random values
        """
        if not hidden_layer:
            self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
            self.b = np.zeros((1, output_dim))
            self.epsilon = epsilon
            self.reg_lambda = epsilon
            self.hidden_layer = False
        else:
            #the structure of the neural net must change in this case
            #let in_h denote input to hidden layer
            self.W_in_h = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
            self.b_in_h = np.zeros((1, hidden_dim))
            #let h_out denote hidden layer to output
            self.W_h_out = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
            self.b_h_out = np.zeros((1, output_dim))
            self.epsilon = epsilon
            self.reg_lambda = 0.01 #can be modified
            self.hidden_layer = True
        
    #--------------------------------------------------------------------------
    
    def compute_cost(self,X, y):
        """
        Computes the total loss on the dataset
        """
        num_samples = len(X)
        if not self.hidden_layer:
            # Do Forward propagation to calculate our predictions
            z = X.dot(self.W) + self.b
            exp_z = np.exp(z)
            softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            # Calculate the cross-entropy loss
            cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
            data_loss = np.sum(cross_ent_err)
            # Add regulatization term to loss
            data_loss += self.reg_lambda/2 * (np.sum(np.square(self.W)))
            return 1./num_samples * data_loss
        else:
            #Forward prop
            z_in = X.dot(self.W_in_h) + self.b_in_h
            #Access the sigmoid function
            activation = 1./(1 + np.exp(-z_in))
            #Let activation denote a new input, acts like X to the hidden layer
            z_out = activation.dot(self.W_h_out) + self.b_h_out
            exp_z = np.exp(z_out)
            softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            cross_ent_err = -np.log(softmax[range(num_samples), y])
            data_loss = np.sum(cross_ent_err)
            # Add regulatization term to loss
            data_loss += self.reg_lambda/2 * (np.sum(np.square(self.W_h_out)))
            return 1./num_samples * data_loss
    
    #--------------------------------------------------------------------------
 
    def predict(self,x):
        """
        Makes a prediction based on current model parameters
        """
        # Do Forward Propagation
        if not self.hidden_layer:
            z = x.dot(self.W) + self.b
            exp_z = np.exp(z)
            softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return np.argmax(softmax, axis=1)
        else:
            #Do the same computation as above, but account for the hidden layer
            z_in = x.dot(self.W_in_h) + self.b_in_h
            activation = 1./(1 + np.exp(-z_in)) 
            z_out = activation.dot(self.W_h_out) + self.b_h_out
            exp_z = np.exp(z_out)
            softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return np.argmax(softmax, axis=1)
            
    #--------------------------------------------------------------------------
    
    def fit(self,X,y,num_epochs):
        """
        Learns model parameters to fit the data
        """
        if not self.hidden_layer:
            for i in range(num_epochs):
                # Do Forward propagation to calculate our predictions
                z = X.dot(self.W) + self.b
                exp_z = np.exp(z)
                softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True) #Our prediction probabilities
                
                #Backpropagation
                beta_z = softmax - construct_truth(y)
                dW = np.dot(X.T,beta_z)
                dB = np.sum(beta_z, axis=0, keepdims=True)
                
                # Add regularization term
                dW += self.reg_lambda * self.W
                
                #Follow the gradient descent
                self.W = self.W - epsilon*dW
                self.b = self.b - epsilon*dB
                
        #HIDDEN LAYER CASE        
        else: 
            for i in range(num_epochs):
                #Forward propagation
                z_in = X.dot(self.W_in_h) + self.b_in_h
                #Access the sigmoid function
                activation = 1./(1 + np.exp(-z_in))
                #Let activation denote a new input, acts like X to the hidden layer
                z_out = activation.dot(self.W_h_out) + self.b_h_out
                exp_z = np.exp(z_out)
                softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True) #our prediction probabilities
                   
                #Backpropagation
                beta_z = softmax
                beta_z[range(len(X)), y] -= 1
                beta_hidden = beta_z.dot(self.W_h_out.T) * (activation - np.power(activation,2))
                dW_h_out = (activation.T).dot(beta_z)
                dB_h_out = np.sum(beta_z, axis=0, keepdims=True)
                dW_in_h = np.dot(X.T, beta_hidden)
                dB_in_h = np.sum(beta_hidden, axis=0)
                
                #Optional regularization terms
                dW_h_out += self.reg_lambda * self.W_h_out
                dW_in_h += self.reg_lambda * self.W_in_h
                
                #Follow the gradient descent
                self.W_in_h += -epsilon * dW_in_h
                self.b_in_h += -epsilon * dB_in_h
                self.W_h_out += -epsilon * dW_h_out
                self.b_h_out += -epsilon * dB_h_out
                  
        return self
        

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

def plot_decision_boundary(pred_func):
    """
    Helper function to print the decision boundary given by model
    """
    # Set min and max values
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

#Train Neural Network on
linear = False

#A. linearly separable data
if linear:
    #load data
    X = np.genfromtxt('/Users/RyanJosephLader/Desktop/Lab4_Soln/DATA/ToyLinearX.csv', delimiter=',')
    y = np.genfromtxt('/Users/RyanJosephLader/Desktop/Lab4_Soln/DATA/ToyLinearY.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()
#B. Non-linearly separable data
else:
    #load data
    X = np.genfromtxt('/Users/RyanJosephLader/Desktop/Lab4_Soln/DATA/ToyMoonX.csv', delimiter=',')
    y = np.genfromtxt('/Users/RyanJosephLader/Desktop/Lab4_Soln/DATA/ToyMoonY.csv', delimiter=',')
    y = y.astype(int)
    #plot data
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()

input_dim = 2 # input layer dimensionality
output_dim = 2 # output layer dimensionality

# Gradient descent parameters 
epsilon = 0.01
num_epochs = 5000

# Fit model
#----------------------------------------------
#Uncomment following lines after implementing NeuralNet
#----------------------------------------------
NN = NeuralNet(input_dim, False, 0, output_dim, epsilon)
NN.fit(X,y,num_epochs)
#
# Plot the decision boundary
plot_decision_boundary(lambda x: NN.predict(x))
plt.title("Neural Net Decision Boundary")
print('Cost of Neural Net: \n')
print(NN.compute_cost(X,y))

correctCount = 0.
predictions = NN.predict(X)
for i in range(len(y)):
    if predictions[i] == y[i]:
        correctCount += 1.0
accuracy = correctCount/len(y)

print('\nAccuracy of Neural Net:\n')
print(accuracy)

for i in range(1,11):
    NN = NeuralNet(input_dim, True, i, output_dim, epsilon)
    NN.fit(X,y,num_epochs)
    print('Cost of Neural Net with ' + str(i) + ' hidden layers: \n')
    print(NN.compute_cost(X,y))
    
    correctCount = 0.
    predictions = NN.predict(X)
    for j in range(len(y)):
        if predictions[j] == y[j]:
            correctCount += 1.0
    accuracy = correctCount/len(y)
    print('\nAccuracy of Neural Net with ' + str(i) + ' hidden layers:\n')
    print(accuracy)
    
