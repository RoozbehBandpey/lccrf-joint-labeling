import nltk
from collections import Counter
from terminaltables import AsciiTable





def build_data_structure(predicted_postag, predicted_chunk, stat):
    if stat == True:
        print("Loading", nltk.corpus.conll2000.fileids()[0], ", and,", nltk.corpus.conll2000.fileids()[1])
    train_sents = list(nltk.corpus.conll2000.iob_sents('train.txt'))
    test_sents = list(nltk.corpus.conll2000.iob_sents('test.txt'))

    # train_dataset = []
    train_token_list = []
    train_POS_tag_list = []
    train_chunk_tag_list = []

    # test_dataset = []
    test_token_list = []
    test_POS_tag_list = []
    test_chunk_tag_list = []

    for sentence in train_sents:
        for triple in sentence:
            train_token_list.append(triple[0])
            train_POS_tag_list.append(triple[1])
            train_chunk_tag_list.append(triple[2])

    for sentence in test_sents:
        for triple in sentence:
            test_token_list.append(triple[0])
            test_POS_tag_list.append(triple[1])
            test_chunk_tag_list.append(triple[2])

    train_token_frequency = Counter(train_token_list)
    train_POS_tag_frequency = Counter(train_POS_tag_list)
    train_NPS_tag_frequency = Counter(train_chunk_tag_list)

    test_token_frequency = Counter(test_token_list)
    test_POS_tag_frequency = Counter(test_POS_tag_list)
    test_NPS_tag_frequency = Counter(test_chunk_tag_list)

    table_data = []
    table_data.append(['Data Statistics', 'Training Data', 'Test Data'])
    table_data.append(['Number of sentences', str(len(train_sents)), str(len(test_sents))])
    table_data.append(['Number of tokens', str(len(train_token_list)), str(len(test_token_list))])
    table_data.append(
        ['POS Tag count', str(len(train_POS_tag_frequency.keys())), str(len(test_POS_tag_frequency.keys()))])
    table_data.append(
        ['Chunk Tag count', str(len(train_NPS_tag_frequency.keys())), str(len(test_NPS_tag_frequency.keys()))])
    table_data.append(
        ['Vocabulary Size', str(len(train_token_frequency.keys())), str(len(test_token_frequency.keys()))])

    table = AsciiTable(table_data)
    if stat == True:
        print(table.table)



    if predicted_postag != None:
        pos_repalced_train_sents = []
        for sents, postag_seq in zip(train_sents, predicted_postag):
            pos_repalced_sents = []
            for triple, postag in zip(sents, postag_seq):
                triple = list(triple)
                triple[1] = postag
                triple = tuple(triple)
                pos_repalced_sents.append(triple)
            pos_repalced_train_sents.append(pos_repalced_sents)


        return pos_repalced_train_sents, test_sents

    if predicted_chunk != None:
        chunk_repalced_train_sents = []
        for sents, chunk_seq in zip(train_sents, predicted_chunk):
            chunk_repalced_sents = []
            for triple, chunk in zip(sents, chunk_seq):
                triple = list(triple)
                triple[2] = chunk
                triple = tuple(triple)
                chunk_repalced_sents.append(triple)
            chunk_repalced_train_sents.append(chunk_repalced_sents)

        return chunk_repalced_train_sents, test_sents

    if predicted_chunk == None and predicted_postag == None:
        return train_sents, test_sents


def POS_word2features(sent, i):
    # Feature construnctor for pos tagger without chunk label features
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features

def POS_word2features_with_chunk(sent, i):
    #Feature construnctor for pos tagger with chunk label features
    word = sent[i][0]
    chunk = sent[i][2]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': chunk
    }
    if i > 0:
        word1 = sent[i-1][0]
        chunk1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': chunk1
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        chunk1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': chunk1
        })
    else:
        features['EOS'] = True

    return features

def Chunk_word2features(sent, i):
    # Feature construnctor for chunker with postag features
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1
        })
    else:
        features['EOS'] = True

    return features


def Chunk_word2features_without_POS(sent, i):
    # Feature construnctor for pos tagger without chunk label features
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper()
        })
    else:
        features['EOS'] = True

    return features


def POS_sent2features(sent):
    return [POS_word2features(sent, i) for i in range(len(sent))]

def POS_sent2features_with_chunk(sent):
    return [POS_word2features_with_chunk(sent, i) for i in range(len(sent))]

def Chunk_sent2features(sent):
    return [Chunk_word2features(sent, i) for i in range(len(sent))]

def Chunk_sent2features_without_POS(sent):
    return [Chunk_word2features_without_POS(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def sent2postag(sent):
    return [postag for token, postag, label in sent]


def load_POS_data(predicted_chunk, stat):
    if predicted_chunk == None:
        train_sents, test_sents = build_data_structure(None, None,stat)
    else:
        train_sents, test_sents = build_data_structure(None, predicted_chunk, stat)

    X_train = [POS_sent2features(s) for s in train_sents]
    y_train = [sent2postag(s) for s in train_sents]


    X_test = [POS_sent2features(s) for s in test_sents]
    y_test = [sent2postag(s) for s in test_sents]

    return X_train, y_train, X_test, y_test


def load_POS_data_with_chunk(predicted_chunk, stat):
    if predicted_chunk == None:
        train_sents, test_sents = build_data_structure(None, None,stat)
    else:
        train_sents, test_sents = build_data_structure(None, predicted_chunk, stat)

    X_train = [POS_sent2features_with_chunk(s) for s in train_sents]
    y_train = [sent2postag(s) for s in train_sents]


    X_test = [POS_sent2features_with_chunk(s) for s in test_sents]
    y_test = [sent2postag(s) for s in test_sents]

    return X_train, y_train, X_test, y_test


def load_Chunk_data(predicted_posttags, stat):
    if predicted_posttags == None:
        train_sents, test_sents = build_data_structure(None, None, stat)
    else:
        train_sents, test_sents = build_data_structure(predicted_posttags, None, stat)

    X_train = [Chunk_sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]


    X_test = [Chunk_sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    return X_train, y_train, X_test, y_test


def load_Chunk_data_without_POS(stat):
    train_sents, test_sents = build_data_structure(None, None, stat)

    X_train = [Chunk_sent2features_without_POS(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]


    X_test = [Chunk_sent2features_without_POS(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    return X_train, y_train, X_test, y_test

