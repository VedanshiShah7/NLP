# STEP 1: rename this file to textclassify_model.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import random
import numpy as np
import sys
from collections import Counter
import math


"""
Your name and file comment here:
Vedanshi Shah & Byron Pham

WE WORKED ON ALL THE PARTS OF THE ASSIGNMENT TOGETHER EXCEPT THE KFOLD CROSS VALIDATION PART AND THE MULTICLASS METRICS FOR PRECISION, RECALL AND F1. VEDANSHI WORKED ON THAT PART.
"""


"""
Cite your sources here:
dataset for multi class: https://www.kaggle.com/code/omkarsabnis/sentiment-analysis-on-the-yelp-reviews-dataset/data 

"""

"""
Implement your functions that are not methods of the TextClassify class here
"""


def generate_tuples_from_file(training_file_path):
    """
    Generates tuples from file formated like:
    id\ttext\tlabel
    Parameters:
      training_file_path - str path to file to read in
    Return:
      a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    """
    f = open(training_file_path, "r", encoding="utf8")
    listOfExamples = []
    for review in f:
        if len(review.strip()) == 0:
            continue
        dataInReview = review.split("\t")
        for i in range(len(dataInReview)):
            # remove any extraneous whitespace
            dataInReview[i] = dataInReview[i].strip()
        t = tuple(dataInReview)
        listOfExamples.append(t)
    f.close()
    return listOfExamples


def precision(gold_labels, predicted_labels):
    """
    Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double precision (a number from 0 to 1)
    """
    # Precision = TruePositives / (TruePositives + FalsePositives)
    true_pos = 0
    false_pos = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and predicted_labels[i] == '1':
            true_pos += 1
        elif gold_labels[i] == '0' and predicted_labels[i] == '1':
            false_pos += 1

    if true_pos == 0 and false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    return precision


def recall(gold_labels, predicted_labels):
    """
    Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double recall (a number from 0 to 1)
    """
    # Recall = TruePositives / (TruePositives + FalseNegatives)
    true_pos = 0
    false_neg = 0
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and predicted_labels[i] == '1':
            true_pos += 1
        elif gold_labels[i] == '1' and predicted_labels[i] == '0':
            false_neg += 1
    if true_pos == 0 and false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)
    return recall


def f1(gold_labels, predicted_labels):
    """
    Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
    Parameters:
        gold_labels (list): a list of labels assigned by hand ("truth")
        predicted_labels (list): a corresponding list of labels predicted by the system
    Returns: double f1 (a number from 0 to 1)
    """
    # F-Measure = (2 * Precision * Recall) / (Precision + Recall)
    precision_val = precision(gold_labels, predicted_labels)
    recall_val = recall(gold_labels, predicted_labels)
    if precision_val + recall_val == 0:
        f1 = 0
    else:
        f1 = (2 * precision_val * recall_val) / (precision_val + recall_val)
    return f1


def precision_multiclass(gold_labels, classified_labels):
    """ Gold labels is a list of strings of the true labels
        Classified labels is a list of strings of the labels assigned by the classifier
        Returns the precision as a float

    Args:
        gold_labels (list): a list of labels assigned by hand ("truth")
        classified_labels (list): a corresponding list of labels predicted by the system

    Returns:
        float: precision from 0 to 1
    """
    unique_labels = set(gold_labels + classified_labels)
    precisions = []

    for label in unique_labels:
        precisions.append(len([i for i in range(len(gold_labels)) if (
            gold_labels[i] == classified_labels[i] and gold_labels[i] == label)])/classified_labels.count(label))

    if len(unique_labels) == 0:
        multi_precision = 0
    else:
        multi_precision = sum(precisions) / len(unique_labels)

    return multi_precision


def recall_multi(gold_labels, classified_labels):
    """gold labels is a list of strings of the true labels
        classified labels is a list of strings of the labels assigned by the classifier
        returns the recall as a float

    Args:
        gold_labels (list): a list of labels assigned by hand ("truth")
        classified_labels (list): a corresponding list of labels predicted by the system

    Returns:
        float: recall as a float
    """
    unique_labels = set(gold_labels + classified_labels)
    recalls = []
    for label in unique_labels:
        recalls.append(len([i for i in range(len(gold_labels)) if (
            gold_labels[i] == classified_labels[i] and gold_labels[i] == label)])/gold_labels.count(label))
    if len(unique_labels) == 0:
        multi_recall = 0
    else:
        multi_recall = sum(recalls) / len(unique_labels)
    return multi_recall


def f1_multi(gold_labels, classified_labels):
    """gold labels is a list of strings of the true labels
        classified labels is a list of strings of the labels assigned by the classifier
        returns the f1 as a float

    Args:
        gold_labels (list): a list of labels assigned by hand ("truth")
        classified_labels (list): a corresponding list of labels predicted by the system

    Returns:
        float: f1 as a float
    """
    precision = precision_multiclass(gold_labels, classified_labels)
    recall = recall_multi(gold_labels, classified_labels)
    multi_f1 = 2 * ((precision * recall) / (precision + recall))
    return multi_f1


"""
Implement any other non-required functions here
"""


"""
implement your TextClassify class here
"""


class TextClassify:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.words_0 = Counter()
        self.words_1 = Counter()

        self.prior_0 = 0
        self.prior_1 = 0

        self.word_data = {0: self.words_0, 1: self.words_1}
        self.vocab = set()

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """
        # calculate prior for each class
        count_0 = 0
        count_1 = 0
        for example in examples:
            if example[2] == "0":
                count_0 += 1
            else:
                count_1 += 1

        self.prior_0 = count_0 / len(examples)
        self.prior_1 = count_1 / len(examples)

        # update bag of words counts in self.word_data
        # word_data format:
        # { 0: Counter(), 1: Counter() }

        for example in examples:
            words = example[1].split()
            self.vocab.update(words)
            if example[2] == '0':
                self.words_0.update(Counter(words))
            else:
                self.words_1.update(Counter(words))

        self.word_data = {'0': self.words_0, '1': self.words_1}

    def score(self, data):
        """
        Score a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        """
        word_probs = {'0': 1, '1': 1}

        word_list = data.split()

        for word in word_list:
            if word in self.words_0:
                word_probs['0'] *= (self.words_0[word] + 1) / \
                    (sum(self.words_0.values()) + len(self.vocab))
            elif word in self.vocab:
                word_probs['0'] *= 1 / \
                    (sum(self.words_0.values()) + len(self.vocab))

            if word in self.words_1:
                word_probs['1'] *= (self.words_1[word] + 1) / \
                    (sum(self.words_1.values()) + len(self.vocab))
            elif word in self.vocab:
                word_probs['1'] *= 1 / \
                    (sum(self.words_1.values()) + len(self.vocab))

        # multiply these by the prior
        word_probs['0'] *= self.prior_0
        word_probs['1'] *= self.prior_1

        return word_probs

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """

        score = self.score(data)
        if score['0'] > score['1']:
            return '0'
        elif score['1'] > score['0']:
            return '1'
        else:
            return '0'

    def featurize(self, data):
        """
        we use this format to make implementation of your TextClassifyImproved model more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """

        data_list = data.split()
        return [(d, True) for d in data_list]

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


def k_fold(all_examples, k):
    """"
    all examples is a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    containing all examples from the train and dev sets

    @return a list of lists containing k sublists where each sublist is one "fold" in the given data
    """
    # Shuffle the examples randomly
    random.shuffle(all_examples)

    # Calculate the size of each fold
    fold_size = len(all_examples) // k

    # Initialize a list to hold the folds
    folds = []

    # Split the examples into k folds
    for i in range(k):
        # Calculate the start and end indices for the current fold
        start_index = i * fold_size
        end_index = (i + 1) * fold_size

        # Create the current fold by taking a slice of the shuffled examples
        current_fold = all_examples[start_index:min(
            len(all_examples), end_index)]

        # Add the current fold to the list of folds
        folds.append(current_fold)

    return folds


class TextClassifyImproved:
    # count(positive words), count(negative words), 'no' in str,
    # count of 1st and 2nd pronouns, if ! doc, log of length

    def __init__(self):
        self.lexicon = self.read_lexicon('vader_lexicon.txt')
        self.weights = []

    def read_lexicon(self, filepath) -> dict:
        output = {}

        with open(filepath, 'r') as f:
            for line in f:
                l = line.split('\t')
                output[l[0]] = l[1]

        return output

    def train(self, examples):
        """
        Trains the classifier based on the given examples
        Parameters:
          examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
        Return: None
        """

        docs = [feat[1] for feat in examples]
        labels = [feat[2] for feat in examples]

        # featurize the training data
        docs_featurized = [[f[1]
                            for f in self.featurize(doc)] for doc in docs]

        # create a vocabulary from the training data
        # set of all unique words in the training data
        vocab = set()
        for doc in docs:
            vocab.update(doc.split())

        y = [1 if label == '1' else 0 for label in labels]

        # train the model with logistic regression
        theta_train = self.train_logistic_regression(docs_featurized, y)

        self.weights = theta_train

    def train_logistic_regression(self, x, y, learning_rate=0.001, num_epoch=100):
        # initialise
        n_features = len(x[0])
        theta = np.zeros(n_features)

        # perform gradient descent
        for epoch in range(num_epoch):
            for i in range(len(x)):
                x_i = x[i]  # features
                y_i = y[i]  # label
                h_i = 1/(1 + np.exp(-np.dot(theta, x_i)))  # score
                gradient = np.multiply((h_i - y_i), x_i)
                theta = theta - learning_rate * gradient

        return theta

    def score(self, data):
        """
        Score a given piece of text
        you will compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
        Parameters:
          data - str like "I loved the hotel"
        Return: dict of class: score mappings
        return a dictionary of the values of P(data | c)  for each class,
        as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
        """
        features = self.featurize(data)
        assert (len(self.weights) == len(features))
        feature_vals = [f[1] for f in features]

        dot_prod = np.dot(self.weights, feature_vals)
        score = 1 / (1 + math.exp(-1 * dot_prod))
        return score

    def classify(self, data):
        """
        Label a given piece of text
        Parameters:
          data - str like "I loved the hotel"
        Return: string class label
        """
        prob = self.score(data)
        return '0' if prob <= 0.5 else '1'

    def featurize(self, data):
        """
        we use this format to make implementation of this class more straightforward and to be
        consistent with what you see in nltk
        Parameters:
          data - str like "I loved the hotel"
        Return: a list of tuples linking features to values
        for BoW, a list of tuples linking every word to True [("I", True), ("loved", True), ("it", True)]
        """
        # features = [pos_words, neg_words, num_nos,
        #             num_1_and_2, num_exclam, log_length]

        features = [0, 0, 0, 0, 0, 0]

        features[5] = math.log(len(data.split()))
        features[4] = data.count('!')

        for word in data.split():
            if word in self.lexicon:
                if float(self.lexicon[word]) > 0:
                    features[0] += 1
                elif float(self.lexicon[word]) < 0:
                    features[1] += 1
            if word.lower() == 'no':
                features[2] += 1
            if word.lower() in [
                'me', 'i', 'my', 'myself', 'mine',
                'we', 'us', 'our', 'ourselves', 'ours',
                'you', 'your', 'yourself', 'yourselves'
            ]:
                features[3] += 1

        return [
            ('pos_words', features[0]),
            ('neg_words', features[1]),
            ('num_nos', features[2]),
            ('num_1_2_pronouns', features[3]),
            ('num_exclm', features[4]),
            ('log_length', features[5]),
            ('bias', 1)
        ]

    def __str__(self):
        return "Logistic Regression Classifier"

    def describe_experiments(self):
        s = """
    Description of your experiments and their outcomes here.
    
    To train our logistic regression model, we used a labeled set of positive and negative movie reviews. The size of the training set was 1600 samples, with 804 positive reviews and 796 negative reviews. To supplement the training data, we also incorporated the VADER sentiment lexicon distributed by NLTK. This word list includes a large set of words pre-labeled as positive, negative, and neutral. Our model featurized each sample as follows: count of positive words, count of negative words, count of “no”s in the document, count of first and second pronouns, number of exclamation marks, and the log of the word count of the document. A bias feature is also included.

    Naive Bayes and Logistic Regression works well on text data because it can capture the complex relationships between the words and labels well on top of handling the high-dimensional feature space. However, it is important to consider the data quality, feature engineering and hyperparameters. For the Bacefook International’s data, it is important to evaluate the performance of any models trained on other datasets before applying them to the data so that we can conduct thorough analysis of the data. This would involve selecting appropriate features and labels and tuning the model hyperparameters to optimize performance on Bacefook’s data.

    Based on this, we chose our epochs of 100 and learning rate of 0.001.
    """
        return s


def main():

    training = sys.argv[1]
    testing = sys.argv[2]
    multi_training = 'training_files/yelp.txt'
    mutli_testing = 'training_files/yelp_test.txt'

    classifier = TextClassify()
    print(classifier)
    # do the things that you need to with your base class
    examples_train_base = generate_tuples_from_file(training)
    classifier.train(examples_train_base)
    examples_dev_base = generate_tuples_from_file(testing)
    y_labels = [e[2] for e in examples_dev_base]
    y_pred = []
    for example in examples_dev_base:
        y_pred.append(classifier.classify(example[1]))

    # report precision, recall, f1
    base_precision = precision(y_labels, y_pred)
    base_recall = recall(y_labels, y_pred)
    base_f1 = f1(y_labels, y_pred)

    print(f'Base precision: {base_precision}')
    print(f'Base recall: {base_recall}')
    print(f'Base f1: {base_f1}')

    # -------------------------------------------------------------------------
    # Improved model
    improved = TextClassifyImproved()
    print(improved)

    # do the things that you need to with your improved class
    examples_train = generate_tuples_from_file(training)
    improved.train(examples_train)
    print(f'LEARNED WEIGHTS: {improved.weights}')

    examples_dev = generate_tuples_from_file(testing)
    y_labels = [e[2] for e in examples_dev]
    y_pred = []
    for example in examples_dev:
        y_pred.append(improved.classify(example[1]))

    # report a summary of your experiments/features here
    print(improved.describe_experiments())

    # report final precision, recall, f1 (for your best model)
    # precision = tp / (tp+fp)
    final_precision = precision(y_labels, y_pred)
    # recall = tp / (tp+fn)
    final_recall = recall(y_labels, y_pred)
    # f1 = 2 * (precision * recall) / (precision + recall)
    final_f1 = f1(y_labels, y_pred)

    print(f'FINAL PRECISION: {final_precision}')
    print(f'FINAL RECALL: {final_recall}')
    print(f'FINAL F1: {final_f1}')

    # get the multi class data
    print('MULTI CLASS')
    improved_multi = TextClassifyImproved()
    print(improved_multi)

    # report multi class precision, recall, f1 (for your best model)
    multi_precision = precision_multiclass(multi_training, mutli_testing)
    # recall = tp / (tp+fn)
    multi_recall = recall_multi(multi_training, mutli_testing)
    # f1 = 2 * (precision * recall) / (precision + recall)
    multi_f1 = f1_multi(multi_training, mutli_testing)

    print(f'MULTI CLASS PRECISION: {multi_precision}')
    print(f'MULTI CLASS RECALL: {multi_recall}')
    print(f'MULTI CLASS F1: {multi_f1}')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python textclassify_model.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
