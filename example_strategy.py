import numpy as np


def model_rewards(payoff_sums, num_plays, num_actions, t):
    if t < 5:
        return np.random.uniform(0, 1, num_actions)
    return np.array([payoff_sums[i] / num_plays[i] for i in range(num_actions)])


def apply_policy(reward_dists, num_actions, t):
    means = np.array(reward_dists)
    z_scores = (means - np.mean(means)) / np.std(means)
    dist = np.exp(z_scores)
    dist /= np.sum(dist)
    return np.random.choice(np.arange(num_actions), p=dist)


# alg: int, (int, int -> float) -> generator
# numActions is the number of
# actions, indexed from 0. reward is a function (or callable) accepting as
# input the action and producing as output the reward for that action
def example_bandit(num_actions, reward):
    payoff_sums = [0.0] * num_actions
    num_plays = [1] * num_actions
    rewards = [0] * num_actions

    # initialize empirical sums
    for a in range(num_actions):
        payoff_sums[a] = 0
        yield a, payoff_sums[a], rewards

    t = 0

    while True:  # apply the random strategy

        reward_expectations = model_rewards(payoff_sums, num_plays, num_actions, t)
        action = apply_policy(reward_expectations, num_actions, t)
        # optinal : update all rewards, but not plays
        theReward = reward(action, t)
        num_plays[action] += 1
        payoff_sums[action] += theReward

        yield action, theReward, reward_expectations
        t = t + 1
