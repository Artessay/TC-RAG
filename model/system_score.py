import numpy as np
import torch.nn.functional as F
import torch

def calculate_perplexity_from_logits(output_seq_flat, target_seq_flat):
    """
    Calculate the perplexity given the logits of both output and target sequences.

    :param output_seq_flat: N x D tensor of logits, where N is the number of tokens in the output sequence.
    :param target_seq_flat: N x D tensor of logits, where N is the number of tokens in the target sequence.
    :return: Perplexity value.
    """
    output_seq_flat = torch.tensor(np.array(output_seq_flat))
    target_seq_flat = torch.tensor(np.array(target_seq_flat))

    min_length = min(output_seq_flat.size(0), target_seq_flat.size(0))
    output_probs = output_seq_flat[:min_length, :]
    target_probs = target_seq_flat[:min_length, :]

    # Flatten the tensors and compute the cross-entropy loss
    loss = F.cross_entropy(output_probs, target_probs, reduction='mean')

    # Calculate perplexity
    perplexity = torch.exp(loss)

    return perplexity.item()


def calculate_low_probablity_from_logprobs(logprobs, hyper_threshold=20):
    """
    Calculate the percentage of values in the log-probability matrix that are below the given threshold.

    :param log_probs: N x 1 numpy array of log probabilities
    :param threshold: A scalar threshold value
    :return: Percentage of values below the threshold
    """
    # Ensure log_probs is a numpy array and is flattened to a 1D array
    log_probs = np.array(logprobs).flatten()

    # Count the number of values below the threshold
    count_below_threshold = np.sum(log_probs < hyper_threshold)

    # Calculate the percentage
    percentage_below_threshold = (count_below_threshold / log_probs.size) * 100

    return percentage_below_threshold


def calculate_low_attention_from_attentions(attentions, hyper_threshold=0.6):
    """
    Calculate the percentage of values in the attentions matrix that are below the given threshold.

    :param attentions: N x 1 numpy array of attentions
    :param threshold: A scalar threshold value
    :return: Percentage of values below the threshold
    """
    # Ensure attentions is a numpy array and is flattened to a 1D array
    attentions = np.array(attentions).flatten()

    # Count the number of values below the threshold
    count_below_threshold = np.sum(attentions < hyper_threshold)

    # Calculate the percentage
    percentage_below_threshold = (count_below_threshold / attentions.size) * 100

    return percentage_below_threshold


def calculate_cuct_from_entropy(entropy):
    """
    Calculate the sum of values in the entropy matrix.

    :param entropy: N x 1 numpy array of entropy
    :return: sum of values
    """
    # Ensure attentions is a numpy array and is flattened to a 1D array
    entropy = np.array(entropy).flatten()

    # Count the sum of values
    entropy_sum = np.sum(entropy)

    return entropy_sum.item()