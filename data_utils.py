import glob
import numpy as np
import re


def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    atheism_examples_train = []
    for doc in glob.glob('20news-bydate/20news-bydate-train/alt.atheism/*'):
        with open(doc, 'r') as f:
            atheism_examples_train.append(f.read())
    atheism_examples_test = []
    for doc in glob.glob('20news-bydate/20news-bydate-test/alt.atheism/*'):
        with open(doc, 'r') as f:
            atheism_examples_test.append(f.read())
    graphics_examples_train = []
    for doc in glob.glob('20news-bydate/20news-bydate-train/comp.graphics/*'):
        with open(doc, 'r') as f:
            graphics_examples_train.append(f.read())
    graphics_examples_test = []
    for doc in glob.glob('20news-bydate/20news-bydate-test/comp.graphics/*'):
        with open(doc, 'r') as f:
            graphics_examples_test.append(f.read())

    # Split by words
    x_train = atheism_examples_train + graphics_examples_train
    x_train = [clean_str(sent) for sent in x_train]
    x_test = atheism_examples_test + graphics_examples_test
    x_test = [clean_str(sent) for sent in x_test]

    # Generate labels
    atheism_labels_train = [[0, 1] for _ in atheism_examples_train]
    graphics_labels_train = [[1, 0] for _ in graphics_examples_train]
    y_train = np.concatenate([atheism_labels_train, graphics_labels_train], 0)
    atheism_labels_test = [[0, 1] for _ in atheism_examples_test]
    graphics_labels_test = [[1, 0] for _ in graphics_examples_test]
    y_test = np.concatenate([atheism_labels_test, graphics_labels_test], 0)
    
    return [x_train, y_train, x_test, y_test]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]