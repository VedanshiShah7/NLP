# imports go here
import collections
import random
import sys
import numpy as np

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
        self.n_minus_one_counts = {}  # REMOVE: or None?
        self.vocab = set()
        self.model = None
        # tokenize without nltk
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
            count = collections.Counter(tokens)
            count_dict = dict(count)

            n_minus_1_gram = self.n_gram - 1
            for i in range(len(tokens) - n_minus_1_gram + 1):
                # get the tokens from current to next
                tokens_list_minus_one = tokens[i:i+n_minus_1_gram]
                n_minus_one_gram_word = " ".join(tokens_list_minus_one)
                ngram_minus_one_list.append(n_minus_one_gram_word)

            for i in range(len(tokens) - self.n_gram + 1):
                # get the tokens from current to next
                tokens_list = tokens[i:i+self.n_gram]
                ngram_word = " ".join(tokens_list)
                ngram_list.append(ngram_word)

            n_minus_one_num = dict(collections.Counter(ngram_minus_one_list))
            ngram_num = dict(collections.Counter(ngram_list))

        # for k in count_dict:
        #     if count_dict[k] == 1 and k != self.line_begin and k != self.line_end:
        #         indx = tokens.index(k)
        #         tokens[indx] = self.UNK

        print('this is tokens')
        print(tokens)

        print('this is n-1')
        print(n_minus_one_num)
        for k in list(n_minus_one_num.keys()):
            if n_minus_one_num[k] == 1 and k != self.line_begin and k != self.line_end:
                # remove k from ngram_num
                n_minus_one_num.pop(k)
                # add UNK to ngram_num
                n_minus_one_num[self.UNK] = 1

        print('this is n')

        print(ngram_num)
        for k in list(ngram_num.keys()):
            if ngram_num[k] == 1 and k != self.line_begin and k != self.line_end:
                print(f'tokens{tokens}')
                # indx = tokens.index(k)

                # remove k from ngram_num
                ngram_num.pop(k)
                # add UNK to ngram_num
                ngram_num[self.UNK] = 1

        print('updated ngram')
        print(ngram_num)
        self.n_minus_one_counts = n_minus_one_num
        self.ngram_counts = ngram_num
        self.total_count = count_dict

        # the vocab is the set of all tokens
        print(f'--------- all tokens: {all_tokens}---------')
        self.vocab = all_tokens

        print(self.ngram_counts)

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
        total_words = len(self.vocab)
        vocab_size = len(self.total_count)

        if self.n_gram == 1:
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

        # if self.n_gram == 2:
        else:
            print('in 2')
            for i in range(0, len(tokens)-1):
                tokens_list = tokens[i: i+self.n_gram]
                tokens_list_new = tokens[i: i+self.n_gram-1][0]
                tokens_list_word = " ".join(tokens_list)

                # for token in tokens_list:
                #     if token not in ngram_count:
                #         token = self.UNK

                if tokens_list_word not in ngram_count:
                    tokens_list_word = self.UNK

                # if tokens_list_word not in n_minus_one_count:
                #     tokens_list_word = self.UNK

                if self.is_laplace_smoothing:
                    if (tokens_list_word) not in ngram_count:  # list of things
                        # calculate the probability of the ngram
                        prob = (ngram_count[tokens_list_word] + 1) / \
                            (n_minus_one_count[tokens_list_word] +
                             vocab_size)
                    else:
                        prob = (ngram_count[tokens_list_word] + 1) / \
                            (n_minus_one_count[tokens_list_new] +
                             vocab_size)  # ngram + 1/ngram_mius_1+ 1+ vocab size
                else:
                    if (tokens_list_word) not in ngram_count:
                        prob = 0
                    else:
                        prob = ngram_count[tokens_list_word] / \
                            n_minus_one_count[tokens_list_word]
                score = score * prob
            return score

        # tokens = self.tokenizer.tokenize(sentence)
        # score = 1
        # for i in range(len(tokens) - self.n_gram + 1):
        #     context = tuple(tokens[i:i+self.n_gram-1])
        #     word = tokens[i+self.n_gram-1]
        #     if context in self.ngram_counts:
        #         if word in self.ngram_counts[context]:
        #             score *= self.ngram_counts[context][word]
        #         else:
        #             score *= 0
        #     else:
        #         score *= 0
        # return score

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        # sentence = ""
        # if self.n_gram == 1:
        #     sentence += "<line> "
        #     next_word = random.choices(list(self.unigram_counts.keys()), list(
        #         self.unigram_counts.values()), k=1)[0]
        #     while next_word != "</line>":
        #         if next_word != "<line>":
        #             sentence += next_word + " "
        #         next_word = random.choices(list(self.unigram_counts.keys()), list(
        #             self.unigram_counts.values()), k=1)[0]
        #     sentence += "</line>"
        # else:
        #     for i in range(self.n_gram - 1):
        #         sentence += "<line> "
        #     ngram_list = list(self.ngram_counts.keys())
        #     next_ngram = random.choices(ngram_list, list(
        #         self.ngram_counts.values()), k=1)[0]
        #     while next_ngram[-1] != "</line>":
        #         sentence += " ".join(next_ngram[:-1]) + " "
        #         next_ngram = random.choices(ngram_list, list(self.ngram_counts.values()), k=1, weights=[
        #                                     self.ngram_counts[ngram]/self.ngram_counts[next_ngram[-self.n_gram+1:]] for ngram in ngram_list if ngram[:-1] == next_ngram[-self.n_gram+1:]])[0]
        #     sentence += " ".join(next_ngram[:-1])
        #     for i in range(self.n_gram - 1):
        #         sentence += " </line>"
        # return sentence

        if self.n_gram == 1:
            unigram_dict = self.total_count
            sentence = "<s>"
            tok = "<s>"

            unigram_list = []
            prob_list = []
            for key in unigram_dict:
                unigram_list.append(key)

            unigram_list = set(unigram_list)
            unigram_list = list(unigram_list)

            for unigram in unigram_list:
                prob = self.score(unigram)
                prob_list.append(prob)

            print(unigram_list)
            print(prob_list)

            prob_so_far = 0
            found = 0
            while tok != "</s>":

                w = np.random.choice(unigram_list, p=prob_list)
                tok = w

                sentence = sentence + " " + tok
                if tok == "</s>":
                    return sentence
            if tok == "</s>":
                return sentence

        if self.n_gram == 2:
            ngrams_dict = self.ngram_counts

            sentence = "<s>"

            tok = "<s>"
            while tok != "</s>":
                bigram_list = []
                prob_list = []
                for key in ngrams_dict:
                    first_token = key.split()[0]
                    if tok == first_token:
                        bigram_list.append(key)

                bigram_list = set(bigram_list)
                bigram_list = list(bigram_list)

                for b in bigram_list:
                    p = self.score(b)
                    prob_list.append(p)

                rando = np.random.choice(
                    [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], 1)

                prob_so_far = 0
                found = 0
                w = ""

                for i in range(0, len(prob_list)):
                    prob_so_far = prob_so_far + prob_list[i]
                    if rando <= prob_so_far and not found:
                        w = bigram_list[i].split()[1]
                        found = 1
                    elif i == len(prob_list) - 1:
                        w = bigram_list[0].split()[1]
                        found = 1

                tok = w

                sentence = sentence + " " + tok
                if tok == "</s>":
                    return sentence
            if tok == "</s>":
                sentence = sentence + "</s>"
                return sentence
            return sentence

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

        # sentences = []
        # for i in range(n):
        #     sentence = ""
        #     if self.n_gram == 1:
        #         sentence = self.line_begin + " "
        #         current_token = self.line_begin
        #         while current_token != self.line_end:
        #             next_token = self._sample_next_token(current_token)
        #             sentence += next_token + " "
        #             current_token = next_token
        #         sentence += self.line_end
        #     else:
        #         sentence = self.line_begin * (self.n_gram - 1) + " "
        #         current_tokens = [self.line_begin] * (self.n_gram - 1)
        #         while current_tokens[-1] != self.line_end:
        #             next_token = self._sample_next_token(current_tokens)
        #             sentence += next_token + " "
        #             current_tokens.pop(0)
        #             current_tokens.append(next_token)
        #         sentence += self.line_end * (self.n_gram - 1)
        #     sentences.append(sentence)
        # return sentences

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
        ngram_lm.train(training_path)

        # test it on the testing data
        ngram_lm.test(testing_path)

        # generate some sentences
        sentences = ngram_lm.generate(10)
        for sentence in sentences:
            print(sentence)

    else:
        # code where you compare the different characters
        character = sys.argv[5]
        print("Runnning for", ngram, "model with character", character)


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) < 5:
        print(
            "Usage:", "python lm.py ngram training_file.txt testingfile.txt line_begin [character]")
        sys.exit(1)

    main()
