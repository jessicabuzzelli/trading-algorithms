import numpy as np


def getSpinResult(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def simpleGamblingSim(win_prob, win_threshold=80, max_episodes=1000):
    episode_winnings = 0
    episode = 1
    winnings = np.ones(max_episodes + 1) * win_threshold
    winnings[0] = 0

    while episode_winnings < win_threshold and episode < max_episodes:
        won = False
        bet_amount = 1

        while not won and episode <= max_episodes and episode_winnings < win_threshold:
            won = getSpinResult(win_prob)

            if won is True:
                episode_winnings += bet_amount
                winnings[episode] = episode_winnings

            else:
                episode_winnings -= bet_amount
                bet_amount = bet_amount * 2
                winnings[episode] = episode_winnings

            episode += 1

    return winnings


def realisticGamblingSim(win_prob, allowance=256, win_threshold=80, max_episodes=1000):
    episode = 0
    winnings = np.zeros(max_episodes)
    episode_winnings = 0

    while episode_winnings < win_threshold and episode < max_episodes and episode_winnings > -allowance:
        won = False
        bet_amount = 1

        while not won and episode_winnings < win_threshold and episode < max_episodes and episode_winnings > -allowance:

            if np.random.random() <= win_prob:
                won = True
            else:
                won = False

            if won == True:
                episode_winnings += bet_amount
                winnings[episode] = episode_winnings
                episode += 1
            else:
                episode_winnings -= bet_amount
                winnings[episode] = episode_winnings

                if bet_amount * 2 >= episode_winnings + allowance:
                    bet_amount = episode_winnings + allowance
                else:
                    bet_amount *= 2

                episode += 1

    if episode_winnings >= win_threshold:
        winnings[episode - 1:] = win_threshold
    if episode_winnings <= -allowance:
        winnings[episode - 1:] = -allowance

    return winnings

