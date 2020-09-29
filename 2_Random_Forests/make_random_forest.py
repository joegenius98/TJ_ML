import numpy as np

'''
1. Repeat k times (for k decision trees):
    (a) Draw a bootstrap sample of size n from original training dataset
    (b) Randomly select d features for tree to split on
    (c) Train decision tree (allowing it to split only using the d features you
    previously selected) on boostraped sample
    (d) Save decision tree
'''

'''
    How I will implement pseudocode:
    (a) To draw a boostrap for a NumPy 2D array:
        i. Implement a random number generator from 0 to the len(2D array) or the number of sublists
            Syntax: np.random.randint(0, length_of_2D_array)
        ii. Create a list of i.
            Syntax: "                "(0, len(...), size = int(proportion * len(...)))
    (b) To randomly select d features from bootstrap:
        i. Random number generator from 0 to 2d_array.shape[1]
            Syntax: np.random.randint(0, 2d_array.shape[1])
        ii. List of i. WITHOUT replacement
            Code:
                set_of_feature_ns = set()
                while len(lst_of_feature_ns) < d: # d is a hyperparameter
                    set_of_feature_ns.add(np.random.randint(0, len(2d_arr)))
                slice_with_d = 2d_arr[:, list(set_of_feature_ns)]
    (c) Use make_tree method to make each tree 
    


'''
labels = [0, 1]
class Node:
    def __init__(self, feature_n=None, threshold=None):
        self.feature_n = feature_n
        self.threshold = threshold
        self.children = []
        # self.c = None

    # def add_c(self, c):
    #     # c stands for "class" (in case this node is a leaf)
    #     self.c = c

    def add_child(self, obj):
        self.children.append(obj)

    def __str__(self):
        return f'feature {self.feature_n} <= {self.threshold} \n children: {self.children}'


class Leaf:
    def __init__(self, a_class):
        self.a_class = a_class


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
    for percentile in [20, 40, 60, 80]: # loop through different percentiles to split data by
        thresholds = np.percentile(dataset[:, :-1], percentile, axis = 0)  # thresholds across all features

        for feature in range(len(thresholds)):  # looping through each threshold by index
            threshold = thresholds[feature]  # retrieving threshold value
            l_child, r_child = split_matrix(dataset, feature, threshold)  # left for below thres., right for above thres
            info_gain = information_gain(dataset, l_child, r_child)  # formula to determine which percentile
            # has the best split

            if info_gain > greatest_IG:
                best_left, best_right, best_f, best_t = l_child, r_child, feature, threshold # store current besties
                greatest_IG = info_gain

    if greatest_IG <= 0.05:
        label_counts = np.array([count_labels(dataset, label) for label in labels])
        return Leaf(labels[label_counts.argmax()])

    # print this node (feature, threshold) of the decision tree
    # hint: use depth to indent

    curr_node = Node(best_f, best_t)

    curr_node.add_child(make_tree(best_left, depth + 1))

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


def build_random_forest(dataset, num_trees, bootstrap_prop, num_samples):
    # stores your trees for the random forest
    # this is your model
    trees = []

    # k represents how many trees you have.
    # n represents some proportion to split on (say .6?)
    # d represents the number of features
    # This is the first hyperparameter to tune
    k, n, d = num_trees, bootstrap_prop, num_samples
    d_rows, d_cols  = dataset.shape[0], dataset.shape[1]
    for _ in range(k): 
        # get a submatrix within matrix --> sample from the dataset
        row_nums = np.random.randint(0, d_rows, size = int(n * d_rows)) #selects row indices for bootstrap
        col_nums = np.append(np.random.randint(0, d_cols-1, size = d), d_cols-1) # selects col indices to select features
        bootstrap_samp = dataset[row_nums]
        total_samp = bootstrap_samp[:, col_nums]

        # train with sampled dataset
        trees.append(make_tree(total_samp, 0))

    print(f'Length of trees array: {len(trees)}')    
    return trees


def test_random_forest(trees, x_test):
    y_test = [] # used for output file
    for train_ex in x_test: # loop through each row, cast votes, determine majority vote
        votes = []
        for tree in trees:
            # pass train_ex to each tree
            # append its output in votes
            votes.append(classify(tree, train_ex))

        # figure out what the most frequent output in votes is
        y_test.append(max(votes, key = votes.count))
        # append this to y_test
    
    print(y_test)
    print(len(y_test))
    return y_test

# processing the data 
def get_data_unlabeled(path_to_testing_file, num_cols):
    x = []
    input = open(path_to_testing_file).read().split("\n")
    for _, row in enumerate(input[1:]):
        input_array = row.split(",")
        # if index == 0:
        #     input_array[0] = '4'
        if len(input_array) == num_cols:  # number of features
            x.append(to_num(input_array))
        else:
            print(len(input_array))
    return x


# change name of file to load in data from a different file.
# for training data (with labels)

def to_num(list_of_str_nums):
    # converts list of strings -> list of numbers
    return [float(i) for i in list_of_str_nums]


def get_data_labeled(training_file, num_cols):
    '''
    :return: matrix with format [ [feature1, feature2, ..., label1], [feature1, ... , label2] , ...]
    '''
    ret_mat = []
    input = open(training_file).read().split("\n")[1:]
    for i in input:
        input_array = i.split(",")
        if len(input_array) == num_cols:  # number of features + number of labels
            ret_mat.append(to_num(input_array))
        else:
            print(len(input_array))
    return ret_mat

    # pass array of labels and method will generate output txt


def generate_output_file(y_test):
    with open('out.csv', 'w') as f:
        f.write("id,solution\n")
        for i in range(len(y_test)):
            f.write(str(i + 1) + "," + str(y_test[i]) + "\n")

if __name__ == "__main__":
    train_data = np.array(get_data_labeled('training.csv', 8))
    test_data = get_data_unlabeled('testing.csv', 7)
    print(f'This is the train data\'s shape: {train_data.shape}')
    print(f'This is the test data\'s shape: {(len(test_data), len(test_data[0]))}')
    lst_trees = build_random_forest(train_data, num_trees=64, bootstrap_prop=0.6, num_samples=4)
    predictions = test_random_forest(lst_trees, test_data)
    generate_output_file(predictions)


