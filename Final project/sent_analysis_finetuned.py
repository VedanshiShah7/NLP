from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline
import numpy as np
import evaluate
import torch
import pickle
from datasets import load_dataset
from huggingface_hub import login

DEFAULT_MODEL_PATH = "fine_tuned_sentiment_pipeline.pkl"

class SentAnalysisFineTunedBERT():

   def train(self, data=None):
      try:
         # loads model from online; requires huggingface login
         self.pipe = pipeline("sentiment-analysis", model="joshiarn/my_awesome_model")
         return
      
      # how the model was originally trained, only needed to be done once ever
      except:
         if not data:
            print("No data provided, model still needs to be trained.")
            return
         
         login()

         model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
         imdb = load_dataset("imdb")

         small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(300))])
         small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(30))])

         def preprocess_function(examples):
            return tokenizer(examples["text"], truncation=True)

         tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

         small_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
         small_test_dataset = small_test_dataset.map(preprocess_function, batched=True)

         data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
         

         def compute_metrics(eval_pred):
            load_accuracy = evaluate.load("accuracy")
            load_f1 = evaluate.load("f1")
         
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
            f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
            return {"accuracy": accuracy, "f1": f1}

         training_args = TrainingArguments(
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            save_strategy="epoch",
            output_dir="results",
            logging_dir="logs",
            load_best_model_at_end=True,
            push_to_hub=True,
         )

         trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
         )

         trainer.train()
         print(trainer.evaluate())
         pipe = pipeline(task='text-classification', model=trainer.model, tokenizer=trainer.tokenizer, return_all_scores=True)
         self.pipe = pipe
         self.trainer = trainer
         trainer.push_to_hub()
         pickle.dump(pipe, open(DEFAULT_MODEL_PATH, "wb"))

   def classify(self, data: List[str]):
      return self.pipe(data)

if __name__ == "__main__":
   print(torch.cuda.is_available())
   model = SentAnalysisFineTunedBERT()
   model.train()
   print(model.classify(["I think everything is great", "I think everything is bad"]))
