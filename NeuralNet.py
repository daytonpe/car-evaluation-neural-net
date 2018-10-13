####################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs
#       for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs
#       for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs
#       for connection from layer 2 (second hidden) to layer 3 (output layer)
####################################################################################################


import numpy as np
import pandas as pd


class NeuralNet:
    def __init__(self, train, activation, header=True, h1=4, h2=2):
        self.activation = activation
        np.random.seed(1)
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train, names=["buying",
                                              "maint",
                                              "doors",
                                              "persons",
                                              "lug_boot",
                                              "safety",
                                              "class"])
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)

        # Find number of input and output layers from the dataset
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers =
        # (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    def __activation(self, x, activation="sigmoid"):
        if self.activation == "sigmoid":
            self.__sigmoid(self, x)
        if self.activation == "tanh":
            self.__tanh(self, x)
        if self.activation == "relu":
            self.__relu(self, x)

    def __activation_derivative(self, x, activation="sigmoid"):
        if self.activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if self.activation == "tanh":
            self.__tanh_derivative(self, x)
        if self.activation == "relu":
            self.__relu_derivative(self, x)

    # tanh and derivative
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_derivative(self, x):
        return 1.0 - np.tanh(x)**2

    # ReLU and derivative
    def __relu(self, x):
        return np.maximum(x, 0)

    def __relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def preprocess(self, raw_input):
        print(raw_input.shape)
        train_set = pd.get_dummies(raw_input)
        # train_set = pd.get_dummies(raw_input, columns=["buying",
        #                                                "maint",
        #                                                "doors",
        #                                                "persons",
        #                                                "lug_boot",
        #                                                "safety"])
        # # TODO make this more eloquent
        # for index, row in train_set.iterrows():
        #     if row['class'] == 'unacc':
        #         train_set.at[index, 'class'] = 0.
        #     elif row['class'] == 'acc':
        #         train_set.at[index, 'class'] = 1.
        #     elif row['class'] == 'good':
        #         train_set.at[index, 'class'] = 2.
        #     elif row['class'] == 'vgood':
        #         train_set.at[index, 'class'] = 3.
        #     else:
        #         row['class'] = -1.  # TODO might be a better way to do this
        # pd.set_option('display.max_columns', 500)
        print(train_set.shape)
        np.savetxt('panda.txt', train_set, delimiter="\t")
        return train_set

    # Below is the training function

    def train(self, activation, max_iterations=1000, learning_rate=0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            # print(error.sum())
            self.backward_pass(out)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("\nAfter " + str(max_iterations)
              + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers):\n")
        print('\n**  w01  **')
        print(self.w01)
        print('\n**  w12  **')
        print(self.w12)
        print('\n**  w23  **')
        print(self.w23)

    def forward_pass(self, test_flag=False):
        # pass our inputs through our neural network
        if test_flag:
            print('testing')

        if self.activation == "sigmoid":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif self.activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif self.activation == "relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out

    def backward_pass(self, out):
        # pass our inputs through our neural network
        self.compute_output_delta(out)
        self.compute_hidden_layer2_delta()
        self.compute_hidden_layer1_delta()

    def compute_output_delta(self, out):
        if self.activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if self.activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if self.activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self):
        if self.activation == "sigmoid":
            delta_hidden_layer2 = (
                self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))

        if self.activation == "tanh":
            delta_hidden_layer2 = (
                self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        if self.activation == "relu":
            delta_hidden_layer2 = (
                self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self):
        if self.activation == "sigmoid":
            delta_hidden_layer1 = (
                self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))

        if self.activation == "tanh":
            delta_hidden_layer1 = (
                self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        if self.activation == "relu":
            delta_hidden_layer1 = (
                self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self):
        if self.activation == "sigmoid":
            delta_input_layer = np.multiply(
                self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))

        if self.activation == "tanh":
            delta_input_layer = np.multiply(
                self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        if self.activation == "relu":
            delta_input_layer = np.multiply(
                self.__relu_derivative(self.X01), self.delta01.dot(self.w01.T))

        self.delta01 = delta_input_layer

    # Implement the predict function for applying the trained model on the  test dataset.
    # Assume that the test dataset has the same format as the training dataset
    # Output the test error from this function

    def predict(self, test, header=True):
        raw_input = pd.read_csv(test)
        test_dataset = self.preprocess(raw_input)
        print(self.y.shape)
        out = self.forward_pass()
        error = 0.5 * np.power((out - self.y), 2)
        print('Total error for test data is', np.sum(error))
        return 0


if __name__ == "__main__":
    neural_network = NeuralNet('train.csv', 'relu')
    print('\n\n***TRAINING***\n')
    neural_network.train('relu')
    print('\n\n***TESTING***\n')
    testError = neural_network.predict('test.csv')
