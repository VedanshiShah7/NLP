import math
from nltk.stem.porter import *
from nltk.corpus import stopwords
import spacy
from datasets import load_dataset

"""
Based on code from HW3
"""


"""
Implement your functions that are not methods of the TextClassify class here
"""

def precision(gold_labels, predicted_labels):
  """
  Calculates the precision for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double precision (a number from 0 to 1)
  """
  num = 0
  denom = 0
  for i in range(len(gold_labels)):
    if gold_labels[i] == '1' and predicted_labels[i] == '1':
      num += 1
      denom += 1
    if gold_labels[i] == '0' and predicted_labels[i] == '1':
      denom += 1
  if denom == 0:
    return float('inf')
  return (num / denom)


def recall(gold_labels, predicted_labels):
  """
  Calculates the recall for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double recall (a number from 0 to 1)
  """
  num = 0
  denom = 0
  for i in range(len(gold_labels)):
    if gold_labels[i] == '1' and predicted_labels[i] == '1':
      num += 1
      denom += 1
    if gold_labels[i] == '1' and predicted_labels[i] == '0':
      denom += 1
  if denom == 0:
    return float('inf')
  return (num / denom)


def f1(gold_labels, predicted_labels):
  """
  Calculates the f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: double f1 (a number from 0 to 1)
  """
  P = precision(gold_labels, predicted_labels)
  R = recall(gold_labels, predicted_labels)
  
  if (P + R == 0):
    return 0
  return (2 * P * R) / (P + R)


def all_score(gold_labels, predicted_labels):
  """
  Prints the precision, recall, and f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: None (prints all)
  """
  print('Precision:\t', precision(gold_labels, predicted_labels))
  print('Recall:\t\t', recall(gold_labels, predicted_labels))
  print('F1:\t\t', f1(gold_labels, predicted_labels))


def precision_multiclass(gold_labels, classified_labels):

  # gold labels is a list of strings of the true labels
  # classified labels is a list of strings of the labels assigned by the classifier
  #return the precision as a float

  totalClass = {}
  correctClass = {}
  for i in range(len(gold_labels)):
    if gold_labels[i] == classified_labels[i]:
      if gold_labels[i] not in correctClass:
        correctClass[gold_labels[i]] = 0
      correctClass[gold_labels[i]] += 1
    
    if classified_labels[i] not in totalClass:
      totalClass[classified_labels[i]] = 0
    totalClass[classified_labels[i]] += 1

  total = 0
  totalPrecision = 0
  for c in totalClass:
    total += 1
    if c not in correctClass:
      continue
    totalPrecision += (correctClass[c] / totalClass[c])

  return totalPrecision / total


def recall_multi(gold_labels, classified_labels):
  # gold labels is a list of strings of the true labels
  # classified labels is a list of strings of the labels assigned by the classifier
  #return the recall as a float
  # if gold[i] == predict[i] -> add 1 to class correct predict
  # total[gold[i]] += 1

  totalClass = {}
  correctClass = {}
  for i in range(len(gold_labels)):
    if gold_labels[i] == classified_labels[i]:
      if gold_labels[i] not in correctClass:
        correctClass[gold_labels[i]] = 0
      correctClass[gold_labels[i]] += 1
    
    if gold_labels[i] not in totalClass:
      totalClass[gold_labels[i]] = 0
    totalClass[gold_labels[i]] += 1

  total = 0
  totalPrecision = 0
  for c in totalClass:
    total += 1
    if c not in correctClass:
      continue
    totalPrecision += (correctClass[c] / totalClass[c])
    
  return totalPrecision / total

 
def f1_multi(gold_labels, classified_labels):
  # gold labels is a list of strings of the true labels
  # classified labels is a list of strings of the labels assigned by the classifier
  #return the f1 score as a float
  P = precision_multiclass(gold_labels, classified_labels)
  R = recall_multi(gold_labels, classified_labels)

  if (P + R == 0):
    return 0
  return (2 * P * R) / (P + R)

def all_score_multi(gold_labels, predicted_labels):
  """
  Prints the precision, recall, and f1 for a set of predicted labels give the gold (ground truth) labels.
  Parameters:
      gold_labels (list): a list of labels assigned by hand ("truth")
      predicted_labels (list): a corresponding list of labels predicted by the system
  Returns: None (prints all)
  """
  print('Precision:\t', precision_multiclass(gold_labels, predicted_labels))
  print('Recall:\t\t', recall_multi(gold_labels, predicted_labels))
  print('F1:\t\t', f1_multi(gold_labels, predicted_labels))

"""
Implement any other non-required functions here
"""



class TextClassify:

  def __init__(self):
    self.logprior = {}
    self.vocab = set()
    self.V = 0
    self.loglikelihood = {}
    self.K = 1
    self.C = 0
    self.ERR = str(self) + " has not been trained!"
    self.neg = True
    

  def train(self, examples):
    """
    Trains the classifier based on the given examples
    Parameters:
      examples - a list of tuples of strings formatted [(id, example_text, label), (id, example_text, label)....]
    Return: None
    """
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    new_ex = []
    for item in examples:
      tokStr = " ".join([token.lemma_ for token in nlp(item[1]) if token.lemma_ not in stop_words])
      if self.neg:
        tokStr = self.negate(tokStr)
      #tokStr = " ".join([word for word in tokStr.split() if word not in stop_words])
      new_ex.append((item[0], tokStr, item[2]))

    examples = new_ex
    totalDocs = 0
    classToNum = {}
    classToWord = {}
    classToWordTotal = {}
    vocab = set()

    # rezip input for higher efficiency (organized by class in D)
    
    for txt in examples:
      totalDocs += 1
      label = txt[2]
      words = txt[1]

      if label not in classToNum:
        classToNum[label] = 0
        classToWord[label] = {}
        classToWordTotal[label] = 0

      classToNum[label] += 1

      for word in words.split():
        vocab.add(word)
        if word not in classToWord[label]:
          classToWord[label][word] = 0
        classToWord[label][word] += 1
        classToWordTotal[label] += 1
      
    self.C = [c for c in classToNum]
    self.logprior = {c:math.log(classToNum[c]/totalDocs) for c in classToNum}
    self.vocab = vocab
    self.V = len(vocab)
    self.loglikelihood = {c:{word:math.log((classToWord[c].get(word, 0) + self.K) / (classToWordTotal[c] + self.V)) for word in self.vocab} for c in classToWord}


  def score(self, data):
    """
    Score a given piece of text
    you'll compute e ^ (log(p(c)) + sum(log(p(w_i | c))) here
    
    Parameters:
      data - str like "I loved the hotel"
    Return: dict of class: score mappings
    return a dictionary of the values of P(data | c)  for each class, 
    as in section 4.3 of the textbook e.g. {"0": 0.000061, "1": 0.000032}
    """
    if not self.vocab:
      raise ValueError(self.ERR)
    
    if self.neg:
      data = self.negate(data)

    test = data.split()
    scoreMap = {}
    for c in self.C:
      classLikelihood = 0
      for word in test:
        if word not in self.vocab:
          continue
        classLikelihood += self.loglikelihood[c][word]
      classLikelihood += self.logprior[c]
      scoreMap[c] = math.exp(classLikelihood)
    return scoreMap

  def classify(self, data):
    """
    Label a given piece of text
    Parameters:
      data - str like "I loved the hotel"
    Return: string class label
    """
    if self.neg:
      data = self.negate(data)
    scoreMap = self.score(data)
    return max(scoreMap, key=scoreMap.get)
  
  def negate(self, data):
    """
    negates the string as discussed in chapter 4 of textbook
    Parameters:
      data - str like "I didn't love the hotel"
    Return: str with negated words (ex. I didn't NOT_love NOT_the NOT_hotel)
    """
    data = data.split()
    newText = []
    negate = False
    negations = {"not", "no", "never"}
    punctuation = {",", ".", "!", "?", ";", ":", "(", ")"}
    for word in data:
      if negate:
        word = 'NOT_' + word
      for punc in punctuation:
        if punc in word:
          negate = False
      if word in negations or (len(word) > 2 and word[-3] == "n't"):
        negate = True
      newText.append(word)
    return " ".join(newText)

  def __str__(self):
    return "Modified HW3 Naive Bayes Classifier"


def main():
  imdb = load_dataset("imdb")

  small_train_dataset = imdb["train"].shuffle(seed=4120).select([i for i in list(range(3000))])
  small_test_dataset = imdb["test"].shuffle(seed=4120).select([i for i in list(range(300))])

  small_train_dataset = [(str(i), item[0], str(item[1])) for i, item in enumerate(zip(small_train_dataset["text"], small_train_dataset["label"]))]
  small_test_dataset = [(str(i), item[0], str(item[1])) for i, item in enumerate(zip(small_test_dataset["text"], small_test_dataset["label"]))]

  sentiment_analysis = TextClassify()
  sentiment_analysis.train(small_train_dataset)

  gold = [x[2] for x in small_test_dataset]
  predicted = [sentiment_analysis.classify(x[1]) for x in small_test_dataset]
  all_score(gold, predicted)


if __name__ == "__main__":
  main()