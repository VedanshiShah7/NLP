import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras import Input
from keras.models import Model, load_model
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Bidirectional, concatenate, SpatialDropout1D
import matplotlib.pyplot as plt
from seqeval.metrics import classification_report

class NERLSTM:
    def train(self,  data: str="ner_dataset.csv", model_path: str="ner_lstm_model"):
        sentences, words, chars, tags = self.get_info_from_data(data)
        n_sentences, n_words, n_chars, n_tags = len(sentences), len(words), len(chars), len(tags)
        words = ["PAD", "UNK"] + words
        wordToInd, indToWord = self.ind_tokenize(words)
        tags = ["PAD"] + tags
        tagToInd, indToTag = self.ind_tokenize(tags)
        chars = ["PAD", "UNK"] + chars
        charToInd, indToChar = self.ind_tokenize(chars)

        max_len = 50
        max_len_char = 10

        X_word = self.index_and_pad(sentences, wordToInd, 0)
        y = self.index_and_pad(sentences, tagToInd, 2)
        X_char = self.index_and_pad(sentences, charToInd, char_split=True)

        X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.1, random_state=4120)
        X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.1, random_state=4120)

        # Checks if model already exists
        if os.path.exists(model_path):
            model = load_model(model_path)
            self.model = model

            history=np.load('my_history.npy',allow_pickle='TRUE').item()

            y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])
            y_pred = np.argmax(y_pred, axis=-1)

            def pred2label(pred):
                out = []
                for pred_i in pred:
                    out_i = []
                    for p in pred_i:
                        out_i.append(indToTag[p].replace("PAD", "O"))
                    out.append(out_i)
                return out
            
            pred_labels = list([list(pred) for pred in pred2label(y_pred)])
            test_labels = list([list(pred) for pred in pred2label(y_te)])

            print(classification_report(pred_labels, test_labels))
            return model, pd.DataFrame(history)

        word_in = Input(shape=(max_len,))
        word_emb = Embedding(input_dim=n_words + 2, output_dim=20,
                            input_length=max_len, mask_zero=True)(word_in)
        char_in = Input(shape=(max_len, max_len_char,))
        char_emb = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                input_length=max_len_char, mask_zero=True))(char_in)
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(char_emb)

        x = concatenate([word_emb, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                    recurrent_dropout=0.6))(x)
        out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)
        model = Model([word_in, char_in], out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        
        history = model.fit([X_word_tr,
                        np.array(X_char_tr).reshape((len(X_char_tr), max_len, max_len_char))],
                        np.array(y_tr).reshape(len(y_tr), max_len, 1),
                        batch_size=32, epochs=10, validation_split=0.1, verbose=1)
        
        hist = pd.DataFrame(history.history)
        model.save(model_path)
        self.model = model
        np.save('my_history.npy',history.history)
        y_pred = model.predict([X_word_te, np.array(X_char_te).reshape((len(X_char_te), max_len, max_len_char))])
        y_pred = np.argmax(y_pred, axis=-1)
        print(classification_report(y_te, y_pred))
        
        return model, hist

    def get_info_from_data(self, filename: str="ner_dataset.csv"):
        data = pd.read_csv("ner_dataset.csv", encoding="latin1")
        data = data.fillna(method="ffill")

        words = sorted(set(data["Word"].values))
        chars = sorted(set([w_i for w in words for w_i in w]))
        tags = sorted(set(data["Tag"].values), reverse=True)

        sentences = data.groupby("Sentence #").apply(lambda x: [x["Word"].values.tolist(), x["POS"].values.tolist(), x["Tag"].values.tolist()])
        sentences = [[(sentences[i][0][j], sentences[i][1][j], sentences[i][2][j]) for j in range(len(sentences[i][0]))] for i in range(len(sentences))]
        return sentences, words, chars, tags


    def ind_tokenize(self, tokens: list):
        tokenToInd = {}
        ind = 0
        for token in tokens:
            if token not in tokenToInd:
                tokenToInd[token] = ind
                ind += 1
        indToToken = {ind: token for token, ind in tokenToInd.items()}
        return tokenToInd, indToToken


    def index_and_pad(self, seq: list, tokenToInd: dict, indWithinTuple: int=0, char_split: bool=False, pad_token: str="PAD", max_len: int=50, max_len_char: int=10):
        finalSeq = []
        if not char_split:
            seq = [[tokenToInd[tok[indWithinTuple]] for tok in s] for s in seq]
            for subseq in seq:
                if len(subseq) < max_len:
                    subseq.extend([tokenToInd[pad_token]] * (max_len - len(subseq)))
                else:
                    subseq = subseq[:max_len]
                finalSeq.append(np.array(subseq))
        if char_split:
            for sentence in seq:
                sent_seq = []
                for i in range(max_len):
                    word_seq = []
                    for j in range(max_len_char):
                        if i < len(sentence) and j < len(sentence[i][0]):
                            word_seq.append(tokenToInd.get(sentence[i][0][j]))
                        else:
                            word_seq.append(tokenToInd.get("PAD"))
                    sent_seq.append(word_seq)
                finalSeq.append(np.array(sent_seq))
        return np.array(finalSeq)


if __name__ == "__main__":
    
    modelClass = NERLSTM()
    model, hist = modelClass.train()
   
    plt.style.use("ggplot")
    plt.figure(figsize=(12,12))
    plt.plot(hist["acc"])
    plt.plot(hist["val_acc"])
    plt.show()
