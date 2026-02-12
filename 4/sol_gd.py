"""
solution to numpy exercise in lab #4
by A. Gepperth 2025.
Performs three gradient descent runs with N steps (can be changed) and eps (can be changed as well),
and visualizes the resulting trajectories for different starting points.
"""
import numpy as np;
import matplotlib.pyplot as plt;

# adapt!
N = 30;
eps = 0.1;


# function from exercise #4
def f1(x):
    return x[0] ** 2 + 2 * x[1] ** 2;


def gradf1(x):
    return np.array([2., 4.]) * x;


# sum of two exponentials centere at (-1,0)^T and (1,0)^T
# f(\vec x) = \exp(-(\vec x - (1,0)^T)^2) + \exp(-(\vec x - (-1,0)^T)^2)
def f2(x):
    x1 = x + np.array([1., 0]);
    x2 = x + np.array([-1., 0]);
    return np.exp(-(x1[0]) ** 2 - (x1[1]) ** 2) + np.exp(-(x2[0]) ** 2 - (x2[1]) ** 2)


def gradf2(x):
    x1 = x + np.array([1., 0]);
    x2 = x + np.array([-1., 0]);
    return np.exp(-(x1[0]) ** 2 - (x1[1]) ** 2) * 2 * x1 + np.exp(-(x2[0]) ** 2 - (x2[1]) ** 2) * 2 * x2;


# performs N steps of gd
def do_gd(oldx, N, f, gradf, display=False):
    trajectory = [oldx];
    oldx = oldx + 0.0;  # make a copy so modifying old will not modify trajectory entries
    newx = oldx + 0.0;  # allocate newx with same size as oldx
    for i in range(0, N):
        newx[:] = oldx - eps * gradf(oldx);  # [:] means overwrite array values, not modify newx reference
        if display == True: print(newx);
        trajectory.append(newx);
        oldx[:] = newx;  # same
    return np.array(trajectory);


def plot_trajectory(X, ax, title):  # X is suppsoed to be a np array with N rows and 2 columns
    ax.scatter(X[:, 0], X[:, 1], c="blue");
    ax.scatter([X[0, 0]], [X[0, 1]], c="red");
    ax.set_xlim(-1.5, 1.5);
    ax.set_ylim(-1.5, 1.5);
    ax.set_aspect("equal");
    ax.set_title(title);
    ax.grid();


if __name__ == "__main__":
    fig, ax = plt.subplots(2, 3);

    X1 = do_gd(np.array([0.1, 1.5]), N, f1, gradf1);
    X2 = do_gd(np.array([-0.1, 1.5]), N, f1, gradf1);
    X3 = do_gd(np.array([-1, -1]), N, f1, gradf1, True);
    X4 = do_gd(np.array([0.1, 1.5]), N, f2, gradf2);
    X5 = do_gd(np.array([-0.1, 1.5]), N, f2, gradf2);
    X6 = do_gd(np.array([-1, -1]), N, f2, gradf2, True);

    plot_trajectory(X1, ax[0, 0], "[0.1,1.5]")
    plot_trajectory(X2, ax[0, 1], "[-0.1,1.5]")
    plot_trajectory(X3, ax[0, 2], "[-1,-1]")
    plot_trajectory(X4, ax[1, 0], "[0.1,1.5]")
    plot_trajectory(X5, ax[1, 1], "[-0.1,1.5]")
    plot_trajectory(X6, ax[1, 2], "[-1,-1]")

    plt.show();
