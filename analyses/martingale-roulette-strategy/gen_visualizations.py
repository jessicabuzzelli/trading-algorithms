import matplotlib.pyplot as plt
from martingale import *


def experiment_one(win_prob):
    # FIGURE ONE
    for i in range(10):
        result = simpleGamblingSim(win_prob)
        plt.plot(result)

    axes = plt.gca()
    plt.title("Unlimited Losses Simulation: Net Winnings vs. Bet Number")
    plt.xlabel("Bet")
    plt.ylabel("Net Winnings")
    plt.legend(['Episode {}'.format(x) for x in range(1, 11)])
    plt.savefig('figure1.png')
    plt.cla()
    plt.clf()

    # FIGURE TWO & THREE
    all_results = np.zeros((1000, 1001))

    for i in range(1000):
        all_results[i, :] = simpleGamblingSim(win_prob)

    medians = np.median(all_results, axis=0)
    means = np.mean(all_results, axis=0)
    std_dev = np.std(all_results, axis=0)

    # FIGURE TWO
    plt.plot(means)
    plt.plot(means + std_dev)
    plt.plot(means - std_dev)

    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-256, 100])
    plt.title("Unlimited Losses Simulation: Mean Winnings by Bet Number (n=1000)")
    plt.xlabel("Bet")
    plt.ylabel("Mean Winnings")
    plt.legend(['Mean', '+ Std. Dev.', '- Std. Dev.'])
    plt.savefig('figure2.png')
    plt.cla()
    plt.clf()

    # FIGURE THREE
    plt.plot(medians)
    plt.plot(medians + std_dev)
    plt.plot(medians - std_dev)

    axes = plt.gca()
    axes.set_xlim([0, 300])
    axes.set_ylim([-256, 100])
    plt.title("Unlimited Losses Simulation: Median Winnings by Bet Number (n=1000)")
    plt.xlabel("Bet")
    plt.ylabel("Median Winnings")
    plt.legend(['Median', '+ Std. Dev.', '- Std. Dev.'])
    plt.savefig('figure3.png')
    plt.cla()
    plt.clf()


def experiment_two(win_prob):
    all_results = np.zeros((1000, 1000))

    for i in range(1000):
        all_results[i, :] = realisticGamblingSim(win_prob)

    medians = np.median(all_results, axis=0)
    means = np.mean(all_results, axis=0)
    std_dev = np.std(all_results, axis=0)

    # FIGURE FOUR
    plt.plot(means)
    plt.plot(means + std_dev)
    plt.plot(means - std_dev)

    axes = plt.gca()
    plt.title("Finite Losses Simulation: Mean Winnings by Bet Number (n=1000)")
    plt.xlabel("Bet")
    plt.ylabel("Mean Winnings")
    plt.legend(['Mean', '+ Std. Dev.', '- Std. Dev.'])
    plt.savefig('figure4.png')
    plt.cla()
    plt.clf()

    # FIGURE FIVE
    plt.plot(medians)
    plt.plot(medians + std_dev)
    plt.plot(medians - std_dev)

    axes = plt.gca()
    plt.title("Finite Losses Simulation: Median Winnings by Bet Number (n=1000)")
    plt.xlabel("Bet")
    plt.ylabel("Median Winnings")
    plt.legend(['Median', '+ Std. Dev.', '- Std. Dev.'])
    plt.savefig('figure5.png')
    plt.cla()
    plt.clf()


def main():
    win_prob = 0.47
    np.random.seed(1)

    experiment_one(win_prob)

    experiment_two(win_prob)


if __name__ == "__main__":
    main()
