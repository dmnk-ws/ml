import numpy as np;
import requests;
import os;
import matplotlib.pyplot as plt;


def relu_bad(X):
    """ Very inefficient Python-only solution, avoid this!! """
    res = X + 0.0;
    for y in range(0, X.shape[0]):
        for x in range(0, X.shape[1]):
            res[y, x] = X[y, x] if X[y, x] > 0. else 0.;
    return res;


def relu(X):
    """ efficient solution """
    return (X > 0.).astype("float32") * X;


def relu2(X):
    """ alterrate efficient solution """
    return np.maximum(X, 0);


class LinClass(object):

    ## allocate weight and bias arrays and store ref 2 train data/labels
    def __init__(self, n_in, n_out, X, T):
        self.W = np.ones([n_in, n_out]) * 0.1;
        self.b = np.ones([1, n_out]) * 0.1;
        self.X = X;
        self.T = T;
        self.N = X.shape[0];

    # cross-entropy loss. Slightly inefficient since log(Y) is computed for all elements of Y
    def loss(self, Y, T):
        return -(np.log(Y) * T).sum(axis=1).mean();

    # cross-entropy loss. Very efficient since log is computed only here needed
    def loss2(self, Y, T):
        return -np.log(Y[range(0, Y.shape[0]), np.argmax(T, axis=1)]).mean()

    def dLdb(self, Y, T):
        return np.zeros(self.b.shape);

    # fill in your code here!
    def dLdW(self, X, Y, T):
        return np.zeros(self.W.shape);

    # softmax: Assumption about X: 2D array
    # Straightforward solution, no indices
    def S(self, X):
        E = np.exp(X);
        return E / E.sum(axis=1).reshape(-1, 1);  # -1 means: compute size of this axis!
        # return E / E.sum(axis=1, keepdims = True) ; # works as well

    # model function self.W is really $W^T$
    # self.b is really $\vec b^T$
    def f(self, X):
        return self.S(np.dot(X, self.W) + self.b);

    # performs a single gradient descent step
    # works with any size of X and T
    def train_step(self, X, T, eps):
        Y = self.f(X);
        loss = self.loss(Y, T);
        dLdb = self.dLdb(Y, T);
        dLdW = self.dLdW(X, Y, T);
        self.b -= eps * dLdb;  ## b(i+1) = b(i) - eps * gradL
        self.W -= eps * dLdW;  ## same
        return loss;

    # perform multiple gradient descent steps and display loss. Does it go down??
    def train(self, max_it, eps):
        for it in range(0, max_it):
            print("iut=", it, "loss=", self.train_step(self.X, self.T, eps));


if __name__ == "__main__":

    ## download MNIST if not present in current dir!
    if os.path.exists("./mnist.npz") == False:
        print("Downloading MNIST...");
        fname = 'mnist.npz'
        url = 'http://www.gepperth.net/alexander/downloads/'
        r = requests.get(url + fname)
        open(fname, 'wb').write(r.content)

    ## read it into
    data = np.load("mnist.npz")
    traind = data["arr_0"];
    trainl = data["arr_2"];
    traind = traind.reshape(60000, 784)

    # train the linear classifier with 300 MNIST samples. Try with
    # 1000, 10000, 60000 and see your memory go down the drain :-)
    lc = LinClass(784, 10, traind[0:300], trainl[0:300]);

    # ------------------
    # solutions
    # ....

    print("---relu part");
    A1 = np.array([[-1, 2, 3], [-1, -2., 5]]);
    print(relu_bad(A1))
    print(relu(A1))
    print(relu2(A1))

    print("---softmax part");
    print(lc.S(A1));

    print("---loss part");
    Y = np.array([[0.4, 0.4, 0.2], [0.7, 0.1, 0.2]]);
    T = np.array([[1, 0, 0], [0, 0., 1]]);
    print(lc.loss(Y, T));
    print(lc.loss2(Y, T));

    # test model on first two samples
    print("---model function part");
    print(lc.f(traind[0:2]))

    # matplotlib
    f, axes = plt.subplots(1, 5);
    # enumerate (x) creates an iterator that loops over tuples (i,z) where z is an element of x
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(traind[i].reshape(28, 28));
    plt.show();

    f, axes = plt.subplots(1, 2);
    axes.ravel()[0].scatter(range(0, traind.shape[0]), traind.mean(axis=1));
    axes.ravel()[1].imshow(traind.mean(axis=0).reshape(28, 28));
    plt.show();

    # ------------------

    # the actual training
    # lc.train(500, 0.01) ;

    # simple way of testing the classifier
    # print(lc.f(traind[0:10]).argmax(axis=1),trainl[0:10].argmax(axis=1))





