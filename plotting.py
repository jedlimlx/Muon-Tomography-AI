import matplotlib.pyplot as plt


def plot_voxels(data1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(data1)
    plt.show()
