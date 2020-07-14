# coding = utf-8
import numpy as np
import jieba
import re
import itertools
from collections import Counter
from keras.utils.np_utils import to_categorical
import os
import pickle
# 创建停用词list
# def stopwordslist(filepath):
#     stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
#     return stopwords

#分词及去掉标点符号
def tokenizer(document, stopwords_file):
    tokenizer_document = []
    stopwords = [line.strip() for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
    for text in document:
        content = jieba.cut(text)
        outstr = ""
        for word in content:
            if word not in stopwords:
                outstr+=word
                outstr+=" "
        tokenizer_document.append(outstr)
    return tokenizer_document


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    #清理数据替换无词义的符号
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    # return string.strip().lower()
    # fileSentence = jieba.cut(string)
    string = ' '.join(i for i in string)
    return string.strip()

def loadDataset(config):
    '''
    读取样本数据
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                    trainingSamples样本数据，每一条都是QA对
    :return: word2id, id2word, trainingSamples
    '''


    dataset_path = os.path.join(config.vocab_dir)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
        word2id = data['word2id']
        id2word = data['id2word']

    return word2id, id2word

def uploadDataset(filename, dic):
    '''
    写入词汇
    :param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，

    :return: None
    '''


    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'wb') as handle:
        pickle.dump(dic, handle)  # Warning: If adding something here, also modifying saveDataset
    #     word2id = data['word2id']
    #     id2word = data['id2word']
    #
    # return word2id, id2word


def load_data_and_labels(config, is_training=True):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
	从文件加载MRpolarity数据，将数据拆分成单词并生成标签。返回分离的句子和标签。
    """
    # Load data from files
    if is_training:
        path = config.data_file
    else:
        path = config.test_file
    data_examples = []
    tag_examples = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            try:
                if line:
                    data_examples.append(line.split("\t\t")[1])
                    tag_examples.append(line.split("\t\t")[0])
            except:
                pass

    x_text = [clean_str(sent) for sent in data_examples]#字符过滤，实现函数见clean_str()
    # Generate labels

    tag = ["C32-Agriculture", "C3-Art", "C19-Computer", "C34-Economy", "C31-Enviornment",
               "C7-History", "C38-Politics", "C11-Space", "C39-Sports",
               "C15-Energy", "C16-Electronics", "C17-Communication", "C23-Mine", "C29-Transport",
               "C35-Law", "C36-Medical", "C37-Military", "C6-Philosophy", "C5-Education", "C4-Literature"]

    label2index = dict()
    idx = 0
    for c in tag:
        label2index[c] = idx
        idx += 1

    index_data = []
    if is_training:
        with open("data/label-id.txt", "w", encoding="utf-8") as f:
            f.write(str(label2index))
    for ind in tag_examples:
        index_data.append(label2index[ind])
    labels = to_categorical(index_data, num_classes=config.num_classes)

    return x_text, index_data, labels

def build_dataset(words, n_words):
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

def str_idx(corpus, dic, maxlen, UNK = 3):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X
#创建batch迭代模块
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    #每次只输出shuffled_data[start_index:end_index]这么多
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1# 每一个epoch有多少个batch_size
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
		#每一代都清理数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size #当前batch的索引开始
            end_index = min((batch_num + 1) * batch_size, data_size) # 判断下一个batch是不是超过最后一个数据了
            yield shuffled_data[start_index:end_index]
