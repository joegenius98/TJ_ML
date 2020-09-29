import numpy as np
from os.path import dirname, join
current_dir = dirname(__file__)

# Construct to make the decision tree
class Node:
    def __init__(self, feature_n=None, threshold=None):
        self.feature_n = feature_n
        self.threshold = threshold
        self.children = []
        # self.c = None

    # def add_c(self, c):
    #     # c stands for "class" (in case this node is a leaf)
    def add_child(self, obj):
        self.children.append(obj)

    def __str__(self):
        return f'feature {self.feature_n} <= {self.threshold} \n children: {self.children}'


class Leaf:
    def __init__(self, a_class):
        self.a_class = a_class

# change name of file to load in data from a different file.
# for testing data (no labels)

# classification labels


labels = [0, 1]


# given shell code to pre-process .csv files

def get_data_unlabeled(path_to_testing_file):
    x = []
    input = open(path_to_testing_file).read().split("\n")
    for index, i in enumerate(input[1:]):
        input_array = i.split(",")
        # if index == 0:
        #     input_array[0] = '4'
        if len(input_array) == 9:  # number of features
            x.append(to_int(input_array))
        else:
            print(len(input_array))
    return x


# change name of file to load in data from a different file.
# for training data (with labels)

def to_int(list_of_str_nums):
    return [int(i) for i in list_of_str_nums]


def get_data_labeled(path_to_training_file):
    '''
    :return: matrix with format [ [feature1, feature2, ..., label1], [feature1, ... , label2] , ...]
    '''
    ret_mat = []


    input = open(path_to_training_file).read().split("\n")[1:]
    for i in input:
        input_array = i.split(",")
        if len(input_array) == 10:  # number of features + number of labels
            ret_mat.append(to_int(input_array))
        else:
            print(len(input_array))
    return ret_mat


# pass array of labels and method will generate output txt


def generate_output_file(y_test):
    with open(join(current_dir, 'out.csv'), 'w') as f:
        f.write("id,class\n")
        for i in range(len(y_test)):
            f.write(str(i + 1) + "," + str(y_test[i]) + "\n")


# my code


def count_labels(dataset, label):
    # returns proportion of dataset that has this label
    return dataset[dataset[:, -1] == label].shape[0]


def gini(dataset):
    """
    :param dataset:
    :return: gini impurity measure of dataset
    """
    len_dataset = dataset.shape[0]
    if len_dataset == 0:
        return 0  # to prevent division by zero error (when len of dataset is 0, must mean 0 IG)
    sum = 0
    for label in labels:
        ratio = count_labels(dataset, label) / len_dataset
        sum += ratio ** 2

    return 1 - sum


def split_matrix(matrix, feature, threshold):
    """
    :param matrix: [[feature1, feature2, ... label], ...] -> np.ndarray
    :param feature: the index of the sublist in the matrix we're interested in
    :param threshold: used on the feature to split into left and right childs
    :return: the left child and right child
    """

    return matrix[matrix[:, feature] <= threshold], matrix[matrix[:, feature] > threshold]


def information_gain(dataset, leftchild, rightchild):
    # do calculations
    N_p, N_left, N_right = dataset.shape[0], leftchild.shape[0], rightchild.shape[0]
    return gini(dataset) - (N_left / N_p) * gini(leftchild) - (N_right / N_p) * gini(rightchild)


def make_tree(dataset, depth):
    """
    :param dataset: numpy array representing training dataset
    :param depth:
    :return: the decision tree based on training dataset
    """

    greatest_IG = 0
    best_left, best_right, best_f, best_t = 0, 0, 0, 0
    for percentile in [20, 40, 60, 80]:
        thresholds = np.percentile(dataset[:, :-1], percentile, axis = 0)

        for feature in range(len(thresholds)):
            threshold = thresholds[feature]
            l_child, r_child = split_matrix(dataset, feature, threshold)
            info_gain = information_gain(dataset, l_child, r_child)

            if info_gain > greatest_IG:
                best_left, best_right, best_f, best_t = l_child, r_child, feature, threshold
                greatest_IG = info_gain

    if greatest_IG <= 0.1:
        label_counts = np.array([count_labels(dataset, label) for label in labels])
        return Leaf(labels[label_counts.argmax()])

    # print this node (feature, threshold) of the decision tree
    # hint: use depth to indent
    print("    " * depth + f'if feature_#{best_f} <= {best_t}:')

    curr_node = Node(best_f, best_t)

    curr_node.add_child(make_tree(best_left, depth + 1))

    print("    " * depth + 'else: ')

    curr_node.add_child(make_tree(best_right, depth + 1))
    return curr_node


def print_matrix(matrix):
    """
    :param matrix: a list of lists
    """
    for list in matrix:
        for num in list:
            if num < 10:
                print(str(num) + "    ", end = '')
            else:
                print(str(num) + "   ", end = '')
                
        print('\n')


def classify(tree, row):
    curr_node = tree
    """
    :param tree: the struct holding decisions of decision tree
    :param row: a row of features 1-9 of a dataset
    :return: the class 0 or 1
    """
    while type(curr_node) is Node:
        if row[curr_node.feature_n] <= curr_node.threshold:
            curr_node = curr_node.children[0]
        else:
            curr_node = curr_node.children[1]

    return curr_node.a_class


if __name__ == '__main__':
    data_matrix = get_data_labeled(join(current_dir, 'training.csv'))
    # print_matrix(data_matrix)
    train_data = np.array(data_matrix)


    # todo: run this only when you double-checked your code
    decision_tree = make_tree(train_data, 0)
    print('\n')
    print(decision_tree)
    print()
    # todo: write some ml
    # todo: use model to generate y_test from x_test

    test_data = get_data_unlabeled(join(current_dir, 'testing.csv'))
    y_test = [classify(decision_tree, test_data_row) for test_data_row in test_data]

    print(len(y_test))
    generate_output_file(y_test)
