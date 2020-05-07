from hw2_corpus_tool import *
import pycrfsuite
from utils import *
import argparse


def utterance2features(dialogue, idx):
    utterance = dialogue[idx]
    speaker = utterance.speaker.lower()
    token_pos_list = utterance.pos
    features = []
    try:
        for token_pos in token_pos_list:
            features.append('TOKEN_' + token_pos.token)
            features.append('POS_' + token_pos.pos)
    except TypeError:
        features.append('NO_WORDS')

    if idx > 0:
        if speaker != dialogue[idx - 1].speaker.lower():
            features.append('SPEAKER_CHANGE')
    else:
        features.append('FIRST_UTTERANCE')

    return features


def utterance2label(utterance):
    try:
        act_tag = utterance.act_tag.lower()
    except:
        act_tag = None
    return act_tag


def dialogue2features(dialogue):
    return [utterance2features(dialogue, idx) for idx in range(len(dialogue))]


def dialogue2labels(dialogue):
    return [utterance2label(utterance) for utterance in dialogue]


def tag(tagger, data):
    predicted_tags = [tagger.tag(dialogue2features(dialogue)) for dialogue in data]
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

    print('correctly tagged', correctly_tagged_count, sep=' ')
    print('total tags', total_tags_count, sep=' ')
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

    MODEL_FILE = 'model.crfsuite'



    # Begin train predict on 10-fold CV
    fold = 1
    total_train_acc = 0
    total_dev_acc = 0
    for train, dev in k_fold_split(list(get_data(input_dir)), k=10, seed=1):
        trainer = pycrfsuite.Trainer(verbose=False)
        for dialogue in train:
            x_seq = dialogue2features(dialogue)
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

        print('Tagging train Begins: FOLD ', fold)
        predicted_tags = tag(tagger, train)
        print('Tagging train Ends: FOLD ', fold)
        train_accuracy = accuracy(predicted_tags,
                                  [dialogue2labels(dialogue) for dialogue in train])
        print('Tagging dev Begins: FOLD ', fold)
        predicted_tags = tag(tagger, dev)
        print('Tagging dev Ends: FOLD ', fold)
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
    # END train predict on 10-fold CV

    # writing the predicted tags in output_file
    output = ''
    for one_dialogue_predicted_tags in predicted_tags:
        for predicted_tag in one_dialogue_predicted_tags:
            output += predicted_tag + '\n'
        output += '\n'
    with open(output_file, 'w') as f:
        f.write(output)


