import numpy as np
from matplotlib import pyplot as plt


def main():
    data = np.loadtxt('results/profiling_data.txt',
                      dtype=np.float,
                      delimiter=',')
    cpu = data[:, 0]
    ram = data[:, 1]

    plt.plot(cpu, color='green')
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
