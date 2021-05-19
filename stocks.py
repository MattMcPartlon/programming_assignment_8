from random import choice


def pairs(L, typeCons):
    return [(typeCons(L[i]), typeCons(L[i + 1])) for i in range(0, len(L) - 1, 2)]


def process_line(stockLine):
    tokens = stockLine.split(',')
    return [tokens[0]] + pairs(tokens[1:], float)


def transpose(A):
    column = lambda A, j: [row[j] for row in A]
    return [column(A, j) for j in range(len(A[0]))]


def read_stock_table(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    headers = lines[0].strip().split(',')
    numericalTable = [[headers[0]] + pairs(headers[1:], typeCons=str)] + [process_line(line) for line in lines[1:]]
    preDictTable = transpose(numericalTable)[1:]

    # convert to a dictionary {str: [(float, float)]}
    stock_history_dict = {}
    for single_history in preDictTable:
        ticker = single_history[0][0].split('-')[0]
        stock_history_dict[ticker] = single_history[1:]

    return stock_history_dict


# Compute the payoff of buying $amountToInvest worth of shares at the opening bell,
# and selling as the last trade of the day.
def payoff(stock_table, t, stock, amount_to_invest=1.0):
    openPrice, closePrice = stock_table[stock][t]

    sharesBought = amount_to_invest / openPrice
    amountAfterSale = sharesBought * closePrice

    return amountAfterSale - amount_to_invest


def payoff_graph(table, tickers, cumulative=True, save_loc='./payoff.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    num_rounds = len(table[tickers[0]])

    reward = lambda choice, t: payoff(table, t, choice)
    single_action_rewards = lambda s: np.array([reward(s, t) for t in range(num_rounds)])
    xs = np.array(list(range(num_rounds)))

    ax1 = plt.subplot(111)

    if cumulative:
        plt.title("Cumulative stock rewards over time")
    else:
        plt.title("Stock rewards over time")

    plt.ylabel('Reward')
    plt.xlabel('Day')

    for ticker in tickers:
        if cumulative:
            ax1.plot(xs, np.cumsum(single_action_rewards(ticker)), label=ticker)
        else:
            ax1.plot(xs, single_action_rewards(ticker), label=ticker)

    plt.legend()
    plt.show()
    plt.savefig(save_loc)
