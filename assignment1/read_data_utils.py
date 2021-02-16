import sys, math, random


def read1(data_file, perc):

    with open(data_file, 'r') as f:
        # reading number of features (M) and number of classes (N)
        mn = f.readline().split(" ")
        M = mn[0]
        N = mn[1]

        # reading file lines and calculating number of training examples
        file_lines = f.readlines()
        total_lines = len(file_lines)
        ntrain_examples = math.floor(perc*total_lines)
        # shuffling data matrix
        random.shuffle(file_lines)

        # building train set
        s_train = []
        t_train = []
        for i in range(ntrain_examples):
            data_line = file_lines[i].split(" ")
            s_train.append(data_line[:M])
            t_train.append(data_line[M:])

        # building test set
        s_test = []
        t_test = []
        for j in range(n_train_examples, total_lines):
            data_line = file_lines[j].split(" ")
            s_test.append(data_line[:M])
            t_test.append(data_line[M:])

    return s_train, t_train, s_test, t_test




def read2(data_file):

    with open(data_file, 'r') as f:
        # reading number of features (M) and number of classes (N)
        mn = f.readline().split(" ")
        M = mn[0]
        N = mn[1]

        # reading file lines and calculating number of training examples
        file_lines = f.readlines()

        # building train/test set
        s = []
        t = []
        for line in file_lines:
            data_line = line.split(" ")
            s.append(data_line[:M])
            t.append(data_line[M:])

    return s, t



def read3(train_file, test_file):

    with open(train_file, 'r') as f:
        # reading number of features (M) and number of classes (N)
        mn = f.readline().split(" ")
        M = mn[0]
        N = mn[1]

        # reading file lines and calculating number of training examples
        file_lines = f.readlines()

        # building train set
        s_train = []
        t_train = []
        for line in file_lines:
            data_line = line.split(" ")
            s_train.append(data_line[:M])
            t_train.append(data_line[M:])

    with open(test_file, 'r') as f:
        # reading number of features (M) and number of classes (N)
        mn = f.readline().split(" ")
        M = mn[0]
        N = mn[1]

        # reading file lines and calculating number of test examples
        file_lines = f.readlines()

        # building test set
        s_test = []
        t_test = []
        for line in file_lines:
            data_line = line.split(" ")
            s_test.append(data_line[:M])
            t_test.append(data_line[M:])

    return s_train, t_train, s_test, t_test
