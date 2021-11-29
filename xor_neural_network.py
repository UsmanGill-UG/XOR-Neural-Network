# Usman Shaukat Gill
# BSCS19010
# XOR Neural Network
import numpy as np


# activation function, we are using is sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# init function
def Initialize_Parameters(input_f, num_of_neurons, output_f):
    w1 = np.random.rand(num_of_neurons, input_f)  # 2 x2 # hidden layers weights
    w2 = np.random.rand(output_f, num_of_neurons)  # 2 x 1 # weight going to final neutron
    b1 = np.zeros((num_of_neurons, 1))  # 2 x 1  # hidden layer bias
    b2 = np.zeros((output_f, 1))  # 1 x 1  # bias at last neutron
    return w1, w2, b1, b2


# forward feed
def Forward_Propagation(X, Y, w1, w2, b1, b2):
    z1 = np.dot(w1, X) + b1  # hidden layer
    z1_hat = sigmoid(z1)  # hidden layer with sigmoid
    z2 = np.dot(w2, z1_hat) + b2  # z2 is final output/layer
    z2_hat = sigmoid(z2)  # final output with sigmoid
    return z1, z1_hat, z2, z2_hat


def Back_Propagation(z1, z1_hat, z2, z2_hat, output_f, w1, w2):
    error_5 = z2_hat * (1 - z2_hat) * (output_f - z2_hat)  # error in output  # 1X4 error matrix
    error_hidden_layer = z1_hat * (1 - z1_hat) * (np.dot(w2.T, error_5))  # error in hidden layer # w2 1x2
    return error_5, error_hidden_layer


# updating weights and Bias
def UpdateWeightsAndBias(z1_hat, z2_hat, w1, w2, b1, b2, e5, e_hidden, lr, X):
    w2 = w2 + (lr * (np.dot(e5, z1_hat.T)))
    w1 = w1 + (lr * (np.dot(e_hidden, X.T)))
    b2 = b2 + lr * (np.dot(e5, np.ones((4, 1))))
    b1 = b1 + lr * (np.dot(e_hidden, np.ones((4, 1))))
    return w1, w2, b1, b2


def main():
    input_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # input
    output_xor = np.array([[0, 1, 1, 0]])  # output
    # print(input_xor)
    # input_xor.shape()  # giving error : 'tuple' object is not callable
    num_of_neurons = 2
    input_f = input_xor.shape[0]
    output_f = output_xor.shape[0]
    w1, w2, b1, b2 = Initialize_Parameters(input_f, num_of_neurons, output_f)
    lr = 0.25
    epochs = 10000  # num of steps
    for i in range(epochs):
        z1, z1_hat, z2, z2_hat = Forward_Propagation(input_xor, output_xor, w1, w2, b1, b2)
        # print("z1 hat : ", z1_hat)
        # print("z2_hat : ", z2_hat)
        e5, e_hidden = Back_Propagation(z1, z1_hat, z2, z2_hat, output_xor, w1, w2)  # Compute gradients /derivatives
        # print("Error 5 : ", e5)
        # print("Error Hidden : ", e_hidden)
        w1, w2, b1, b2 = UpdateWeightsAndBias(z1_hat, z2_hat, w1, w2, b1, b2, e5, e_hidden, lr,
                                              input_xor)  # Update weights & bias

    o = Forward_Propagation(input_xor, output_xor, w1, w2, b1, b2)
    o = o[3]
    print("Without Threshold : ", o)
    o[o < 0.5] = 0
    o[o >= 0.5] = 1
    print("With Threshold : ", o)


if __name__ == '__main__':
    main()
