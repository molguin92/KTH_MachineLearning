import numpy as np
from matplotlib import pyplot as plt


def main():
    data = np.loadtxt('results/profiling_data.txt',
                      dtype=np.float,
                      delimiter=',')
    cpu = data[:, :-1]
    ram = data[:, -1]

    f, axes = plt.subplots(3, 3, sharex='col', sharey='row')

    for i in range(9):
        x = int(i / 3)
        y = i % 3
        axes[x, y].plot(cpu[:, i])

    plt.ylabel('CPU%')
    plt.xlabel('Time [s]')
    plt.show()

    # plt.plot(ram)
    ram = ram / np.power(10, 9)
    plt.fill_between(range(len(ram)), ram, color='orange')
    plt.ylabel('RAM [GB]')
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':
    main()
