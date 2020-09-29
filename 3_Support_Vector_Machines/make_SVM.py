from sklearn import svm

# Data Processing
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

