import numpy as np


# activation function, we are using is sigmoid, returns 1 or 0
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Initialize_Parameters(input_f, num_of_neurons, output_f):
    w1 = np.random.randn(num_of_neurons, input_f)  # 2 x2
    w2 = np.random.randn(output_f, num_of_neurons)  # 2 x 1
    b1 = np.zeros((num_of_neurons, 1))  # 2 x 1
    b2 = np.zeros((output_f, 1))  # 1 x 1
    print("w1 : ", w1)
    print("w2 : ", w2)
    print("b1 : ", b1)
    print("b2 : ", b2)
    return w1, w2, b1, b2


def Forward_Propagation(X, Y, w1, w2, b1, b2):
    z1 = np.dot(w1, X) + b1  # hidden layer
    z1_hat = sigmoid(z1)
    z2 = np.dot(w2, z1_hat) + b2  # z2 is final output/layer
    z2_hat = sigmoid(z2)
    mse = np.mean(np.square(Y - z2_hat))  # mean square error
    return mse, z1, z1_hat, z2, z2_hat


def Back_Propagation():
    lol = 0


def main():
    input_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # input
    output_xor = np.array([[0, 1, 1, 0]])  # output
    # print(input_xor)
    # input_xor.shape()  # giving error : 'tuple' object is not callable
    num_of_neurons = 2
    input_f = input_xor.shape[0]
    output_f = output_xor.shape[0]
    w1, w2, b1, b2 = Initialize_Parameters(input_f, num_of_neurons, output_f)
    epoch = 5000
    lr = 0.01
    total_loss = np.zeros((epoch, 1))
    mse, z1, z1_hat, z2, z2_hat = Forward_Propagation(input_xor, output_f, w1, w2, b1, b2)
    print("z1 : ", z1)
    print("z1 hat : ", z1_hat)
    print("z2 : ", z2)
    print("z2_hat : ", z2_hat)
    print("mse : ", mse)


if __name__ == '__main__':
    main()
