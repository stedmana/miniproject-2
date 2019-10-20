import csv
import validation
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize


def csv_to_list(file_name):
    list_out = []
    cols = 0
    with open(file_name) as csv_to_read:
        csv_reader = csv.reader(csv_to_read, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names for {file_name} are {", ".join(row)}')
                cols = len(row)
                line_count += 1
            else:
                # if len(row) == cols:
                print
                list_out.append(row)
                line_count += 1
        print(f'Processed {len(list_out)} lines.')
    # list_out = remove_quotes(list_out)
    return list_out

# def remove_quotes(list_in):
#     for l_item in list_in:
#         l_item[1] = l_item[1][:]
#     return list_in


def pipeline_main():
    train = csv_to_list('reddit_train.csv')
    tr = len(train)
    test = csv_to_list('reddit_test.csv')
    ts = len(test)
    sets = k_folds(5, train)
    # for x in sets:
    #     print('part 1: {}   part 2: {}'.format(len(x[0]), len(x[1])))
    print(f'train: {tr}, test: {ts}')
    for i, set in enumerate(sets):
        print(f'0  {len(set[0])}     1  {len(set[1])}')
        x = set[0]
        a = np.array(x)
        train = a[:, 1:]
        y = set[1]
        a_1 = np.array(y)
        test = a_1[:, 1:]
        fl_nm_train = 'train_kfold_' + str(i) + '.csv'
        fl_nm_test = 'test_kfold_' + str(i) + '.csv'
        train_df = pd.DataFrame(train)
        test_df = pd.DataFrame(test)
        train_df.to_csv(fl_nm_train)
        test_df.to_csv(fl_nm_test)
        erer = 54
        # validator_pipeline(set, train_stub, eval_stub)
    # validator_pipeline(train, )
    # trainer(train, print)


def train_stub(train_on):
    # train_on is a list. [0] is the comment, [1] is the subreddit
    pass


def eval_stub(comment_in):
    return 'hockey'


def k_folds(k, list_in):
    list_out = []
    for i_k in range(k):
        leave_in = []
        leave_out = []
        for index, item in enumerate(list_in):
            if index % k == i_k:
                leave_out.append(item)
            else:
                leave_in.append(item)
        list_out.append([leave_in.copy(), leave_out.copy()])
    return list_out


def validator_pipeline(training_list, train_fn, eval_fn):
    training = training_list[0]
    testing = training_list[1]
    trainer(training, train_fn)
    class_options = ['hockey', 'nba', 'leagueoflegends',
                     'soccer', 'funny', 'movies', 'anime',
                     'Overwatch', 'trees', 'GlobalOffensive',
                     'nfl', 'AskReddit', 'gameofthrones', 'conspiracy',
                     'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
    true_positive = make_dict()
    true_negative = make_dict()
    false_positive = make_dict()
    false_negative = make_dict()
    accuracy = make_dict_float()
    precision = make_dict_float()
    recall = make_dict_float()
    false_positive_rate = make_dict_float()
    f1_measure = make_dict_float()
    # print(true_positive)
    print(f'{len(training)}   {len(testing)}')
    for test_item in testing:
        # print(test_item)
        result = eval_fn(test_item[1])
        expected = test_item[2]
        list_less_result = list_remove_item(class_options, result)
        list_less_expected = list_remove_item(class_options, expected)
        if result == expected:
            for class_item in list_less_expected:
                true_negative[class_item] += 1
            true_positive[result] += 1
        else:
            false_positive[result] += 1
            false_negative[expected] += 1
            list_less_result_expected = list_remove_item(list_less_result, expected)
            for class_item in list_less_result_expected:
                true_negative[class_item] += 1
    # next part computes accuracy, precision, recall... for all classes
    for class_item in class_options:
        tp = true_positive[class_item]
        tn = true_negative[class_item]
        fp = false_positive[class_item]
        fn = false_negative[class_item]
        print(f'tp:{tp}  tn:{tn}  fp:{fp}  fn:{fn}')
        x = tp + tn + fp + fn
        print(f'sum : {x}    expected: {len(testing)}')
        # accuracy[class_item] = (tp + tn) / (tp + fp + tn + fn)  # accuracy = (TP + TN) / (TP + FP + TN + FN)
        # precision[class_item] = tp / (tp + fp)  # Precision = TP / (TP + FP)
        # recall[class_item] = tp / (tp + fn)  # recall = TP / (TP + FN)
        # false_positive_rate[class_item] = fp / (fp + tn)  # false positive rate = FP / (FP + TN)
        # f1_measure[class_item] = \
        #     (2 * precision[class_item] * recall[class_item]) / (precision[class_item] + recall[class_item])
        # F1 measure = 2 * (precision * recall) / (precision + recall)



def list_remove_item(list_in, to_remove):
    list_out = []
    for i in list_in:
        if i != to_remove:
            list_out.append(i)
    return list_out


def make_dict_float():
    l = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
    dict = {}
    for l_i in l:
        dict[l_i] = 0.0
    return dict


def make_dict_list():
    l = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
    dict = {}
    for l_i in l:
        dict[l_i] = []
    return dict


def make_dict():
    l = ['hockey', 'nba', 'leagueoflegends', 'soccer', 'funny', 'movies', 'anime', 'Overwatch', 'trees', 'GlobalOffensive', 'nfl', 'AskReddit', 'gameofthrones', 'conspiracy', 'worldnews', 'wow', 'europe', 'canada', 'Music', 'baseball']
    dict = {}
    for l_i in l:
        dict[l_i] = 0
    return dict
    # for i in list_in:
    #     is_there = False
    #     for j in l:
    #         if j == i[2]:
    #             is_there = True
    #     if not is_there:
    #         l.append(i[2])

def trainer(list_in, train_fn):
    for element in list_in:
        train_fn(element[1:])


# pipeline_main()


def word_looker():
    train = csv_to_list('reddit_train.csv')
    dl = make_dict_list()
    for x in train:
        dl[x[2]].append(x)
    return dl


def append_list(in_list, to_append):
    for i, list_item in enumerate(in_list):
        list_item.append(to_append[i])
    return in_list

pipeline_main()
# x = append_list([[8],[9],[10]], [4,5,6])
# print(x)
# word_looker()