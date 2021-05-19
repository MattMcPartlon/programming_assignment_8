from random import shuffle

from example_strategy import example_bandit
from stats import *
from stocks import *


def _run_experiment(stockTable, bandit, augment_stocks=False):
    # if augment_stocks is set to true, then a stock is added with 0 variance-
    # equivalent to choosing an "empty" action when things are uncertain
    tickers = list(stockTable.keys())
    if augment_stocks:
        if 'empty' not in stockTable:
            stockTable['empty'] = [(1, 1) for _ in range(len(stockTable['amzn']))]
    shuffle(tickers)  # note that this makes the algorithm unstable/randomized
    numRounds = len(stockTable[tickers[0]])
    numActions = len(tickers)

    # the true reward
    reward = lambda choice, t: payoff(stockTable, t, tickers[choice])
    singleActionReward = lambda j: sum([reward(j, t) for t in range(numRounds)])

    bestAction = max(range(numActions), key=singleActionReward)
    bestActionCumulativeReward = singleActionReward(bestAction)

    cumulativeReward = 0
    t = 0
    example_Generator = bandit(numActions, reward)
    for (chosenAction, reward, means) in example_Generator:
        cumulativeReward += reward
        t += 1
        if t == numRounds:
            break

    return cumulativeReward, bestActionCumulativeReward, means, tickers[bestAction]




def run_experiment(table, bandit, n_trials = 10):
    pretty_list = lambda L: ', '.join(['%.3f' % x for x in L])
    payoff_stats = lambda data: get_stats([_run_experiment(data, bandit)[0] for _ in range(n_trials)])
    print("(Expected payoff, variance) over " + str(n_trials) + " trials is %r" % (payoff_stats(table),))
    reward, bestActionReward, expectations, bestStock = _run_experiment(table, bandit)
    print("For a single run: ")
    print("Payoff was %.2f" % reward)
    print("Regret was %.2f" % (bestActionReward - reward))
    print("Best stock was %s at %.2f" % (bestStock, bestActionReward))
    print("expectations: %r" % pretty_list(expectations))


if __name__ == "__main__":
    table = read_stock_table('./fortune-500.csv')
    run_experiment(table, example_bandit)
    payoff_graph(table, list(sorted(table.keys())), cumulative=True)
