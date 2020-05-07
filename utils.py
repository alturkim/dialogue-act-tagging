import random
import math
import string


def train_dev_split(all_data, dev_size, seed=1):
    all_idxs = list(range(len(all_data)))
    dev_len = math.floor(dev_size * len(all_data))
    random.seed(seed)
    dev_idxs = random.sample(all_idxs, dev_len)
    train_idxs = list(set(all_idxs) - set(dev_idxs))
    train_set = [all_data[i] for i in train_idxs]
    dev_set = [all_data[i] for i in dev_idxs]
    return train_set, dev_set


def k_fold_split(all_data, k=10, seed=1):
    data_len = len(all_data)
    if data_len < k:
        print('ERROR: #folds is greater than #samples')
        return None, None
    all_idxs = list(range(data_len))
    random.seed(seed)
    random.shuffle(all_idxs)
    fold_len = data_len // k
    for i in range(k):
        if i == k - 1 and (i + 1) * fold_len < data_len:
            start_slice = i * fold_len
            end_slice = data_len
        else:
            start_slice = i * fold_len
            end_slice = (i + 1) * fold_len

        dev_idxs = all_idxs[start_slice:end_slice]
        train_idxs = list(set(all_idxs) - set(dev_idxs))

        train_set = [all_data[i] for i in train_idxs]
        dev_set = [all_data[i] for i in dev_idxs]
        yield train_set, dev_set


def get_utterance_token_str(utterance):
    try:
        return ' '.join([token_pos.token.lower().translate(str.maketrans('', '', string.punctuation))
                         for token_pos in utterance.pos])
    except TypeError:
        return ''


def get_utterance_bigrams(utterance_tokens_str):
    word_list = utterance_tokens_str.split()
    for idx in range(len(word_list)):
        if idx > 0:
            bigram = word_list[idx - 1] + ' ' + word_list[idx]
        else:
            bigram = '<START>' + ' ' + word_list[idx]
        yield bigram


# def get_utterance_unigrams(utterance_tokens_str):
#     word_list = utterance_tokens_str.split()
#     return word_list


def get_bigrams_freq(data, min_freq):
    bigrams_freq_d = {}
    for dialogue in data:
        for utterance in dialogue:
            utterance_tokens_str = get_utterance_token_str(utterance)
            utterance_bigrams = get_utterance_bigrams(utterance_tokens_str)
            for utterance_bigram in utterance_bigrams:
                try:
                    bigrams_freq_d[utterance_bigram] += 1
                except KeyError:
                    bigrams_freq_d[utterance_bigram] = 1

    sorted_bigrams_freq_d = \
        {k: v for k, v in sorted(bigrams_freq_d.items(), key=lambda item: item[1], reverse=True)}
    most_frequent_bigrams_d = {k: v for k, v in sorted_bigrams_freq_d.items() if v >= min_freq}
    return most_frequent_bigrams_d


# def get_unigrams_freq(data, min_freq=1, max_freq=100000):
#     unigrams_freq_d = {}
#     for dialogue in data:
#         for utterance in dialogue:
#             utterance_tokens_str = get_utterance_token_str(utterance)
#             utterance_unigrams = get_utterance_unigrams(utterance_tokens_str)
#             for utterance_unigram in utterance_unigrams:
#                 try:
#                     unigrams_freq_d[utterance_unigram] += 1
#                 except KeyError:
#                     unigrams_freq_d[utterance_unigram] = 1
#
#     sorted_unigrams_freq_d = \
#         {k: v for k, v in sorted(unigrams_freq_d.items(), key=lambda item: item[1], reverse=True)}
#     most_frequent_unigrams_d = {k: v for k, v in sorted_unigrams_freq_d.items() if min_freq <= v <= max_freq}
#     return most_frequent_unigrams_d
