import numpy as np
from collections import defaultdict

class TextClassifyImproved:

    def __init__(self, alpha=0.01, epochs=100, batch_size=32):
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab = set()
        self.classes = set()
        self.word_freq = defaultdict(lambda: defaultdict(int))
        self.class_freq = defaultdict(int)
        self.weights = {}
        
    def train(self, examples):
        self.classes = set(label for _, _, label in examples)
        self.vocab = set(word for example in examples for word, _ in self.featurize(example[1]))
        
        for example in examples:
            text = example[1]
            label = example[2]
            self.class_freq[label] += 1
            
            for word, count in self.featurize(text):
                self.word_freq[label][word] += count
                
        for label in self.classes:
            self.weights[label] = defaultdict(float)
            for word in self.vocab:
                self.weights[label][word] = 0.01
                
        for epoch in range(self.epochs):
            np.random.shuffle(examples)
            
            for i in range(0, len(examples), self.batch_size):
                batch = examples[i:i+self.batch_size]
                gradients = defaultdict(lambda: defaultdict(float))
                
                for example in batch:
                    text = example[1]
                    label = example[2]
                    x = self.featurize(text)
                    y = 1 if label == '1' else 0
                    
                    for word, count in x:
                        gradients[label][word] += (y - self.score(text)[label]) * count
                
                for label in self.classes:
                    for word in self.vocab:
                        self.weights[label][word] += self.alpha * (gradients[label][word] - 0.01 * self.weights[label][word])
                        
    def score(self, data):
        scores = defaultdict(float)
        for label in self.classes:
            scores[label] = sum(self.weights[label][word] * count for word, count in self.featurize(data))
        exp_scores = {label: np.exp(score) for label, score in scores.items()}
        sum_exp_scores = sum(exp_scores.values())
        return {label: exp_scores[label] / sum_exp_scores for label in self.classes}
        
    def classify(self, data):
        scores = self.score(data)
        return max(scores, key=scores.get)
        
    def featurize(self, data):
        words = data.lower().split()
        return [(word, 1) for word in words if word.isalpsha()]
        
    def __str__(self):
        return "Logistic Regression Classifier"
    
    def describe_experiments(self):
        s = """
        We trained a logistic regression classifier on the provided examples using the following hyperparameters:
        - learning rate (alpha): {}
        - number of epochs: {}
        - batch size: {}

        The following evaluation metrics were obtained on the test set:
        - accuracy: {}
        - precision: {}
        - recall: {}
        - F1 score: {}
        """.format(self.alpha, self.epochs, self.batch_size, accuracy, precision, recall,
