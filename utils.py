
import numpy as np
import matplotlib.pyplot as plt


def plot_output(targets, probs, prefix):
    # sort alphabetically
    targets, probs = zip(*sorted(zip(targets, probs), key=lambda tup: tup[0]))
    # transform into array
    probs = np.array(probs)
    x = np.arange(len(targets))

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(111)
    ax.bar(x, probs)
    ax.set_title("History: '{}'".format("".join(prefix)))
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_yticks(np.linspace(0, 1, 9))

    return ax


def entropy(probs, vocab_size):
    probs = np.array(probs)
    if len(probs) < vocab_size:
        probs = np.concatenate([probs, np.zeros(vocab_size - len(probs)) + 1e-6], axis=0)
    return -(probs * np.log(probs)).sum(axis=0)


def apply_temperature(probs, tau):
    """
    Apply temperature and renormalize to output distribution
    Tau is usually between 0 and 1
    """
    new_probs = probs ** (1 / tau)
    return new_probs / new_probs.sum()


def apply_tok_k_sampling(probs, top_k):
    """
    """
    min_prob = probs[probs.argsort()[-top_k]]
    # zero out probabilities less than the probability of top_k element
    new_probs = np.copy(probs)
    new_probs[probs < min_prob] = 0

    return new_probs


def apply_nucleus_sampling(probs, top_p):
    """
    """
    sort_idxs = np.argsort(probs)[::-1]
    sort = probs[sort_idxs]
    cumsum = np.cumsum(sort)
    drop = cumsum > top_p
    # ensure at least one symbol has non-0 probability
    drop[1:] = drop[:-1]
    drop[0] = False
    new_probs = np.copy(probs)
    new_probs[sort_idxs[drop]] = 0

    return new_probs
