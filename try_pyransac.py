from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import random
import pyransac3d as pyrsc


def get_dummy_3d_data():

    plot_data = list()
    for y in range(50):
        for x in range(50):
            plot_data.append(np.array([x, y, 0.1 * x + random.uniform(2.7, 3.8) - 0.1 * y]))


    plot_data = np.asarray(plot_data)
    print(plot_data)
    return plot_data


def plot(plot_data, equation_data):
    x = plot_data[:, 0]
    y = plot_data[:, 1]
    z = plot_data[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    #ax.set_zlim(-5, 20)
    plt.xlabel("X Achse")
    plt.ylabel("Y Achse")

    surf = ax.plot_trisurf(x, y, z, cmap="plasma", linewidth=0)
    fig.colorbar(surf)

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()

    a, b, c, d = equation_data

    x_plane = np.linspace(0, 50, 1000)
    y_plane = np.linspace(0, 50, 1000)

    X, Y = np.meshgrid(x_plane, y_plane)
    Z = (d - a * X - b * Y) / c

    #fig = plt.figure()
    ax_plane = fig.gca(projection='3d')
    ax.set_zlim(-5, 20)

    ax_plane.plot_surface(X, Y, Z)



    ax.view_init(10, 45)
    plt.show()

def make_plane(data):
    #points = load_points(.)  # Load your point cloud as a numpy array (N, 3)
    points = data
    plane1 = pyrsc.Plane()
    best_eq, best_inliers = plane1.fit(points, 0.01)
    return best_eq




data = get_dummy_3d_data()

plane_equation = make_plane(data)


plot(data, plane_equation)
