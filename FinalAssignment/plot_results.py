import numpy as np
import time
from matplotlib import pyplot as plt


def main():
    data = np.genfromtxt('results/profiling_data.txt',
                         dtype=np.float,
                         delimiter=',',
                         names=True)

    f, axes = plt.subplots(3, 3, sharex='col', sharey='row')

    for i in range(9):
        x = int(i / 3)
        y = i % 3

        if i == 4:
            axes[x, y].plot(data['cpu_total'])
            axes[x, y].set_ylabel('CPU%')
            axes[x, y].set_xlabel('Time [s]')
            axes[x, y].set_title('cpu_total')
        else:
            index = 'cpu' + str(i) if i < 4 else 'cpu' + str(i - 1)
            axes[x, y].plot(data[index])
            axes[x, y].set_title(index)

    plt.show()

    # plt.plot(ram)
    ram = data['ram'] / np.power(10, 9)
    plt.fill_between(range(len(ram)), ram, color='orange')
    plt.ylabel('RAM [GB]')
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':
    main()
