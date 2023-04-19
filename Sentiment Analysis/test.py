from textclassify_model import generate_tuples_from_file
from collections import Counter

tuples = generate_tuples_from_file('training_files/movie_reviews_train.txt')
labels = [i[2] for i in tuples]
print(Counter(labels))
