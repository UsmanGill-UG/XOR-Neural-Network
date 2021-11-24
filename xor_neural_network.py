import numpy as np


class perceptron:
    def __init__(self, x, y, xw, yw, b):  # constructor
        self.x = x
        self.y = y
        self.xw = xw
        self.yw = yw
        self.b = b


def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    if res > 0.5:
        return 1
    else:
        return 0


def get_y_val(y):
    #print((y.x * y.xw) + (y.y * y.yw) + y.b)
    return sigmoid((y.x * y.xw) + (y.y * y.yw) + y.b)


def main():
    x = 0
    y = 0
    y1 = perceptron(x, y, 1, 1, -1)
    y1_val = get_y_val(y1)
    y2 = perceptron(x, y, -1, -1, 1)
    y2_val = get_y_val(y2)
    y3 = perceptron(y1_val, y2_val, 1, 1, -2)
    y3_val = get_y_val(y3)
    print("y1 : ", y1_val)
    print("y2 : ", y2_val)
    print("y3 : ", y3_val)


if __name__ == '__main__':
    main()
