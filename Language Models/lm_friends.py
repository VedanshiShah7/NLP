# imports go here
import collections
import random
import sys
import numpy as np
import statistics

# feel free to add imports.

"""
Don't forget to put your name and a file comment here.

Name: Vedanshi Shah

This is an individual homework.
"""


# Feel free to implement more helper functions


"""
Provided helper functions
"""


def read_sentences(filepath):
    """
    Reads contents of a file line by line.
    Parameters:
      filepath (str): file to read from
    Return:
      list of strings
    """
    f = open(filepath, "r")
    sentences = f.readlines()
    f.close()
    return sentences


def get_data_by_character(filepath):
    """
    Reads contents of a script file line by line and sorts into 
    buckets based on speaker name.
    Parameters:
      filepath (str): file to read from
    Return:
      dict of strings to list of strings, the dialogue that speaker speaks
    """
    char_data = {}
    script_file = open(filepath, "r", encoding="utf-8")
    for line in script_file:
        # extract the part between <speaker> tags
        speakers = line[line.index(
            "<speakers>") + len("<speakers>"): line.index("</speakers>")].strip()
        if not speakers in char_data:
            char_data[speakers] = []
        char_data[speakers].append(line)
    return char_data


"""
This is your Language Model class
"""


class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"

    def __init__(self, n_gram, is_laplace_smoothing, line_begin="<line>", line_end="</line>"):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
          line_begin (str): the token designating the beginning of a line
          line_end (str): the token designating the end of a line
        """
        self.line_begin = line_begin
        self.line_end = line_end
        # your other code here
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.ngram_counts = {}
        self.n_minus_one_counts = {}
        self.vocab = set()
        self.model = None
        self.tokenizer = lambda x: x.split()

    def train(self, sentences):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with line_begin and end with line_end
        Parameters:
          sentences (list): list of strings, one string per line in the training file

        Returns:
        None
        """

        ngram_minus_one_list = []
        ngram_list = []
        all_tokens = []

        for sentence in sentences:
            tokens = self.tokenizer(sentence)
            all_tokens += tokens

            n_minus_1_gram = self.n_gram - 1
            for i in range(len(tokens) - n_minus_1_gram + 1):
                tokens_list_minus_one = tokens[i:i+n_minus_1_gram]
                n_minus_one_gram_word = " ".join(tokens_list_minus_one)
                ngram_minus_one_list.append(n_minus_one_gram_word)

            for i in range(len(tokens) - self.n_gram + 1):
                tokens_list = tokens[i:i+self.n_gram]
                ngram_word = " ".join(tokens_list)
                ngram_list.append(ngram_word)

            n_minus_one_num = dict(collections.Counter(ngram_minus_one_list))
            ngram_num = dict(collections.Counter(ngram_list))

        n_1_num_unk = 0

        for k in list(n_minus_one_num.keys()):
            if n_minus_one_num[k] == 1 and k != self.line_begin and k != self.line_end:
                # remove k from ngram_num
                n_minus_one_num.pop(k)
                # add UNK to ngram_num
                n_1_num_unk += 1
            n_minus_one_num[self.UNK] = n_1_num_unk

        num_unk = 0
        for k in list(ngram_num.keys()):
            if ngram_num[k] == 1 and k != self.line_begin and k != self.line_end:
                # remove k from ngram_num
                ngram_num.pop(k)
                # add UNK to ngram_num
                num_unk += 1
            ngram_num[self.UNK] = num_unk

        if ngram_num[self.UNK] == 0:
            ngram_num.pop(self.UNK)

        if n_minus_one_num[self.UNK] == 0:
            n_minus_one_num.pop(self.UNK)

        count = collections.Counter(tokens)
        count_dict = dict(count)
        self.n_minus_one_counts = n_minus_one_num
        self.ngram_counts = ngram_num
        self.total_count = count_dict

        # the vocab is the set of all tokens
        self.vocab = all_tokens

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """

        score = 1
        n_minus_one_count = self.n_minus_one_counts
        ngram_count = self.ngram_counts
        tokens = sentence.split()
        total_words = sum(self.ngram_counts.values())
        vocab_size = len(self.n_minus_one_counts)

        if self.n_gram == 1:
            vocab_size = len(self.ngram_counts)

            for word in tokens:
                if word not in ngram_count:
                    word = self.UNK

                if self.is_laplace_smoothing:
                    prob = (ngram_count[word] + 1) / \
                        (total_words + vocab_size)
                else:
                    prob = ngram_count[word] / total_words
                score = score * prob
            return score

        else:
            for i in range(0, len(tokens)-1):
                tokens_list = tokens[i: i+self.n_gram]
                tokens_list_new = tokens[i: i+self.n_gram-1][0]
                tokens_list_word = " ".join(tokens_list)

                if self.is_laplace_smoothing:
                    if (tokens_list_word) not in ngram_count:
                        prob = (ngram_count.get(tokens_list_word, 0) + 1) / \
                            (n_minus_one_count.get(tokens_list_new, 0) +
                             vocab_size)
                    else:
                        prob = (ngram_count.get(tokens_list_word, 0) + 1) / \
                            (n_minus_one_count.get(tokens_list_new, 0) +
                             vocab_size)
                else:
                    if (tokens_list_word) not in ngram_count:
                        prob = 0
                    else:
                        prob = ngram_count[tokens_list_word] / \
                            n_minus_one_count[tokens_list_new]
                score = score * prob
            return score

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """

        # check if begin is in ngram_dict
        ngram_dict = self.ngram_counts
        begin = self.line_begin
        if begin in ngram_dict:

            if self.n_gram == 1:

                # getting the probability list
                unigram_dict = self.ngram_counts
                # getting the begining value
                begin = self.line_begin
                # setting the current token to begin
                current_token = begin
                sentence = begin

                # storing the value to add back later
                beg_val = unigram_dict[begin]
                # remove line_begin from unigram_dict
                unigram_dict.pop(self.line_begin)

                # create a probability list
                prob_list = []

                # calculate the probability of each unigram
                for unigram in unigram_dict:
                    prob = self.score(unigram)
                    # storing it in the list
                    prob_list.append(prob)

                # converting the dictionary to a list to use in the random function
                unigram_list = list(unigram_dict.keys())

                # sampling the unigram list based on the probability list without the begining token
                while current_token != self.line_end:
                    # using np.random.choice as it is much faster than random.choices
                    current_token = np.random.choice(unigram_list, p=prob_list)
                    # adding the randomly picked current token to the sentence
                    sentence = sentence + " " + current_token

                # adding back the value of line_begin to the unigram_dict
                # so it is not mutated for future calls
                unigram_dict[self.line_begin] = beg_val
                return sentence

            else:
                # getting the probability list
                # n_minus_1_gram_dict = self.n_minus_one_counts
                ngram_dict = self.ngram_counts

                # getting the begining value and setting it as the starting value
                begin = self.line_begin

                sentence = begin

                current_token = self.line_begin

                beg_val = ngram_dict[begin]
                # remove line_begin from unigram_dict
                ngram_dict.pop(self.line_begin)

                # creating a probability list
                prob_list = []

                # calculating the probability of each n minus 1 gram
                for n_gram in ngram_dict:
                    prob = self.score(n_gram)
                    prob_list.append(prob)

                # converting the dictionary to a list to use in the random function
                n_gram_list = list(ngram_dict.keys())

                # creating a counter for line end tokens
                line_end_tks = 0

                while line_end_tks != (self.n_gram - 1):
                    current_token = np.random.choice(n_gram_list, p=prob_list)
                    if current_token == self.line_end:
                        line_end_tks += 1
                    sentence = sentence + " " + current_token

                # adding back the value of line_begin to the unigram_dict
                # so it is not mutated for future calls
                ngram_dict[self.line_begin] = beg_val

                # tacking on the begin tokens to the beginning of the sentence
                for i in range(0, self.n_gram - 1):
                    line_begins = self.line_begin
                    sentence = line_begins + " " + sentence

                return sentence
        else:
            if self.n_gram == 1:
                sentence = '<s> </s>'
            else:
                line_begins = self.line_begin
                line_end = self.line_end
                sentence = line_begins * \
                    (self.n_gram - 1) + " " + line_end * (self.n_gram - 1)

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
            n (int): the number of sentences to generate

        Returns:
            list: a list containing strings, one per generated sentence
        """
        sentences = []

        for i in range(0, n):
            new_sentence = self.generate_sentence()
            sentences.append(new_sentence)

        return sentences

    def perplexity(self, test_sequence):
        """Measures the perplexity for the given test sequence with this trained model. 
        As described in the text, you may assume that this sequence 
        may consist of many sentences "glued together".

        Parameters:
        test_sequence (string): a sequence of space-separated tokens to measure the perplexity of

        Returns:
        float: the perplexity of the given sequence
        """

        tokens = self.tokenizer.tokenize(test_sequence)
        prob = 1.0
        for i in range(len(tokens) - self.n_gram + 1):
            context = tuple(tokens[i:i+self.n_gram-1])
            word = tokens[i+self.n_gram-1]
            if self.is_laplace_smoothing:
                count = self.ngram_counts.get((context, word), 0) + 1
                total_count = sum(self.ngram_counts.get(
                    context, {}).values()) + len(self.vocab)
            else:
                count = self.ngram_counts.get((context, word), 0)
                total_count = sum(self.ngram_counts.get(context, {}).values())
            prob *= count / total_count
        return prob ** (-1 / len(tokens))


def main():
    # TODO: implement the rest of this!
    ngram = int(sys.argv[1])
    training_path = sys.argv[2]
    testing_path = sys.argv[3]
    line_begin = sys.argv[4]
    if len(sys.argv) == 5:
        print("Runnning for", ngram, "model")

        # instantiate a language model like....
        ngram_lm = LanguageModel(
            ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")

        # train it on the training data
        ngram_lm.train(read_sentences(training_path))

        testing_data = read_sentences(testing_path)

        # test it on the testing data
        # ngram_lm.test(read_sentences(testing_path))
        scores = []
        for sentence in testing_data:
            scores.append(ngram_lm.score(sentence))
        print("Number of sentences: {}".format(len(testing_data)))
        print("Average score: {}".format(statistics.mean(scores)))
        print("Std deviation: {}".format(statistics.stdev(scores)))

        # generate some sentences
        sentences = ngram_lm.generate(10)
        for sentence in sentences:
            # scores.append(ngram_lm.score(sentence))
            print(sentence)

    else:
        # code where you compare the different characters
        # character = sys.argv[5]
        # print("Runnning for", ngram, "model with character", character)
        training_data = get_data_by_character(training_path)
        testing_data = get_data_by_character(testing_path)
        models = {}
        for character in training_data:
            if (len(training_data[character]) > 500):
                models[character] = LanguageModel(
                    ngram, True, line_begin="<" + line_begin + ">", line_end="</" + line_begin + ">")
                models[character].train(training_data[character])
        for character in testing_data:
            if (len(testing_data[character]) > 500):
                print("Comparing: {}".format(character))
                bestMatch = ""
                bestScore = 10 ^ 100
                for characterI in models:
                    if (characterI == character):
                        pass
                    score = 0
                    count = len(testing_data[character])
                    for sentence in testing_data[character]:
                        score += models[characterI].score(sentence)
                    score /= count
                    if (score <= bestScore):
                        bestScore = score
                        bestMatch = characterI
                print("Best match: {}".format(bestMatch))
                print("With avg score: {}".format(bestScore))


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) < 5:
        print(
            "Usage:", "python lm.py ngram training_file.txt testingfile.txt line_begin [character]")
        sys.exit(1)

    main()
