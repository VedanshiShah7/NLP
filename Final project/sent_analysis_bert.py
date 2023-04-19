from transformers import pipeline
from load_tweets import load_tweets
from sent_analysis_naive_bayes import all_score

class SentAnalysisBERT():
    def __init__(self):
        self.pipeline = pipeline("sentiment-analysis")

    def predict(self, data):
        predicts = self.pipeline(data)
        predicts = ['0' if x['label']=='NEGATIVE' else '1' for x in predicts]
        return predicts
    
    def score(self, data):
        return self.pipeline(data)

def main():
    data = load_tweets()

    # Training is not performed for this dataset
    #train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]

    model = SentAnalysisBERT()
    predicts = model.predict([x[1] for x in test_data])
    gold = [x[2] for x in test_data]
    all_score(gold, predicts)

if __name__ == "__main__":
    main()