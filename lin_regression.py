import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

N_POINTS = 10


def calculate_plane_equation(data):
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append(np.array([xs[i], ys[i], 1]))
        tmp_b.append(np.array([zs[i]]))

    b = np.asarray(tmp_b)
    c = np.asarray(tmp_A)
    c_transpose = c.T

    # Manual solution
    fit = np.dot(np.dot(np.linalg.inv(np.dot(c_transpose, c)), c_transpose), b)
    errors = np.dot((b - c), fit)
    residual = np.linalg.norm(errors)

    print(errors.shape)

    # Or use Scipy
    # from scipy.linalg import lstsq
    # fit, residual, rnk, s = lstsq(A, b)

    print("solution:")
    print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors:")
    print(errors)
    print("residual:")
    print(residual)

    return fit, residual, errors


def plot(data):
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    # plot raw data
    plt.figure()
    ax = plt.subplot(111, projection='3d')

    font = "Times New Roman"
    ax.set_xlabel("x", fontname=font, fontsize=12)
    ax.set_ylabel("y", fontname=font, fontsize=12)
    ax.set_zlabel("z", fontname=font, fontsize=12)
    ax.set_zlim(-5, 20)
    plt.rcParams["font.family"] = "Comic Sans MS"
    plt.rcParams["font.size"] = 12

    ax.scatter(xs, ys, zs, color='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    x_limit = ax.get_xlim()
    y_limit = ax.get_ylim()
    #x_limit = (2, 5)
    #y_limit = (2, 15)

    plane_data = list()

    for item in data:
        if x_limit[0] <= item[0] <= x_limit[1] and y_limit[0] <= item[1] <= y_limit[1]:
            plane_data.append(item)
    plane_data = np.asarray(plane_data)

    # plot plane
    fit, residual, errors = calculate_plane_equation(plane_data)
    X, Y = np.meshgrid(np.arange(x_limit[0], x_limit[1] + 1),
                       np.arange(y_limit[0], y_limit[1] + 1))

    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r, c] = fit[0] * X[r, c] + fit[1] * Y[r, c] + fit[2]
    ax.plot_wireframe(X, Y, Z, color='k')

    #xlim = np.max([plane_data[:, 0]]), np.min([plane_data[:, 0]])
    #ylim = np.max([plane_data[:, 1]]), np.min([plane_data[:, 1]])+5000

    plt.show()


data = list()
for x in range(N_POINTS):
    for y in range(N_POINTS):
        data.append(np.array([x, y, random.uniform(1, 10) + 0.2 * x + 0.05 * y]))
data = np.asarray(data)

plot(data)
