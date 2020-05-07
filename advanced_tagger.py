from hw2_corpus_tool import *
import pycrfsuite
from utils import *
import argparse


def generate_bigram_features(utterance, frequent_bigrams):
    utterance_token_str = get_utterance_token_str(utterance)
    for utterance_bigram in get_utterance_bigrams(utterance_token_str):
        if utterance_bigram in frequent_bigrams:
            yield utterance_bigram


def utterance2features(dialogue, idx, frequent_bigrams):
    utterance = dialogue[idx]
    speaker = utterance.speaker.lower()
    token_pos_list = utterance.pos
    features = ['BIAS', 'UTTERANCE_LENGTH' + str(len(get_utterance_token_str(utterance)))]

    try:
        for token_pos in token_pos_list:
            features.append('TOKEN_' + token_pos.token)
            if token_pos.token.istitle():
                features.append('TOKEN_ISTITLE')
            if token_pos.token.isdigit():
                features.append('TOKEN_ISDIGIT')
            if token_pos.token.isupper():
                features.append('TOKEN_ISUPPER')
            features.append('POS_' + token_pos.pos)
    except TypeError:
        features.append(utterance.text)

    if idx > 0:
        if speaker != dialogue[idx - 1].speaker.lower():
            features.append('SPEAKER_CHANGE')

        prev_token_pos_list = dialogue[idx - 1].pos
        try:
            for token_pos in prev_token_pos_list:
                features.append('PREV_POS_' + token_pos.pos)
        except TypeError:
            features.append('PREV_' + dialogue[idx - 1].text)

    # elif idx == 0:
    #     features.append('FIRST_UTTERANCE')

    features.extend(list(generate_bigram_features(utterance, frequent_bigrams)))

    return features


def utterance2label(utterance):
    try:
        act_tag = utterance.act_tag.lower()
    except:
        act_tag = None
    return act_tag


def dialogue2features(dialogue, frequent_bigrams):
    return [utterance2features(dialogue, idx, frequent_bigrams) for idx in range(len(dialogue))]


def dialogue2labels(dialogue):
    return [utterance2label(utterance) for utterance in dialogue]


def tag(tagger, data, frequent_bigrams):
    predicted_tags = [tagger.tag(dialogue2features(dialogue, frequent_bigrams)) for dialogue in data]
    return predicted_tags


def accuracy(predicted_tags, correct_tags):
    """
    predicted_tags: list of list of tags
    correct_tags: list of list of tags
    """
    correctly_tagged_count = 0
    total_tags_count = 0
    for one_dialogue_predicted_tags, one_dialogue_correct_tags \
            in zip(predicted_tags, correct_tags):
        assert (len(one_dialogue_predicted_tags)
                == len(one_dialogue_correct_tags)), 'tag lists misalignment'

        total_tags_count += len(one_dialogue_correct_tags)
        for predicted_tag, correct_tag in zip(one_dialogue_predicted_tags, one_dialogue_correct_tags):
            if predicted_tag == correct_tag:
                correctly_tagged_count += 1

    return correctly_tagged_count / total_tags_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUTDIR')
    parser.add_argument('TESTDIR')
    parser.add_argument('OUTPUTFILE')
    args = parser.parse_args()
    input_dir = args.INPUTDIR
    test_dir = args.TESTDIR
    output_file = args.OUTPUTFILE

    MODEL_FILE = 'model_advanced.crfsuite'

    fold = 1
    total_train_acc = 0
    total_dev_acc = 0
    for train, dev in k_fold_split(list(get_data(input_dir)), k=10, seed=1):
        trainer = pycrfsuite.Trainer(verbose=False)

        bigram_frequency_dict = get_bigrams_freq(train, min_freq=30)
        frequent_bigrams = [k for k, v in bigram_frequency_dict.items()]

        for dialogue in train:
            x_seq = dialogue2features(dialogue, frequent_bigrams)
            y_seq = dialogue2labels(dialogue)
            trainer.append(xseq=x_seq, yseq=y_seq)

        trainer.set_params({
            'c1': 1.0,
            'c2': 1e-3,
            'max_iterations': 50,
            'feature.possible_transitions': True
        })

        print('Training Begins: FOLD ', fold)
        trainer.train(MODEL_FILE)
        print('Training Ends: FOLD ', fold)

        tagger = pycrfsuite.Tagger()
        tagger.open(MODEL_FILE)

        print('Tagging Begins: FOLD ', fold)
        predicted_tags = tag(tagger, train, frequent_bigrams)
        train_accuracy = accuracy(predicted_tags,
                                  [dialogue2labels(dialogue) for dialogue in train])
        predicted_tags = tag(tagger, dev, frequent_bigrams)
        print('Tagging Ends: FOLD ', fold)
        dev_accuracy = accuracy(predicted_tags,
                                [dialogue2labels(dialogue) for dialogue in dev])

        print('train accuracy: FOLD ', fold, ',', train_accuracy, sep=' ')
        print('dev accuracy: FOLD ', fold, ',', dev_accuracy, sep=' ')
        total_train_acc += train_accuracy
        total_dev_acc += dev_accuracy
        trainer = None
        tagger = None
        fold += 1
    print('*****************************************')
    print('Average Train Accuracy = ', total_train_acc/10)
    print('Average Dev Accuracy = ', total_dev_acc/10)

    # writing the predicted tags in output_file
    output = ''
    for one_dialogue_predicted_tags in predicted_tags:
        for predicted_tag in one_dialogue_predicted_tags:
            output += predicted_tag + '\n'
        output += '\n'
    with open(output_file, 'w') as f:
        f.write(output)


