# NLP final project

Vedanshi Shah and Arnav Joshi

Topic category: Evaluation

### Overview

**What are you doing? Give us a one-sentence description of what you hope to do in your project.**
We are running 3 models on the text generated from the Whisper data and then comparing the analysis provided from it.
**What is your data? (identify specific data set(s) for this step, link to them and describe them in detail)**

- We will be using the data to train the BERT model, sentiment analysis and the named entity recognition data.
- We will use this dataset for the named entity recognition training and testing with the annotated dataset.
  - Other datasets that we could potentially look at: CoNLL 2003, OntoNotes 5.0.
- We will use this dataset to train and test the sentiment analysis model for the twitter dataset.
  - Other datasets that we could potentially look at: IMDb movie reviews, Amazon product reviews, Yelp reviews.
- Fine-tuning the BERT model - Question Answering: SQuAD (Stanford Question Answering Dataset), TriviaQA.
  **What tools will you be using?**
  We hope to use models and tools like Whisper (Text-to-speech), BERT (language model) and a GPT model (between 3 and 4 presumably GP-3.5), NER, Sentiment analysis, pytorch, nltk and hugging face. For the visualization models we hope to use matplotlib and plotly.

### Models

**What models will you be using?**
We plan on using the Whisper model (and possibly models like the Mozilla DeepSpeech model) alongside models such as BERT, GPT3.5, and manually written neural networks to analyze speech to text data like sentiment analysis or named entity recognition.
**Which of these will you be implementing yourself vs. using models provided by libraries?**
Whisper will be provided by libraries as well as GPT3.5. We will presumably write our NNs ourselves and configure BERT as we desire.
**Identify specific options if you are using models from implemented packages**
The Whisper model has multiple different options for sizes of datasets that it was trained upon. We will attempt to test all three for their performance.

### External

**Any other resources that you’ll need to use? Components you’ll need to implement?**
We will need to implement a way to compare all of the different Whisper models as well as how to compare the output of the BERT model versus our other models.
**What visualizations/results/etc will you be producing?**
We will create an interface to have speech converted to text live and all the 3 models run with the model evaluations. We will
**What are preliminary sources/tutorials/etc that might be helpful?**
Preliminary sources and tutorials that will be useful will probably be the Whisper documentation and OpenAI’s tutorials using it as well as Mozilla’s explanation of how DeepSpeech works (as a fallback for understanding).

### Logistics

**What is your timeline/working plan?**
Week 1 starting March 27, 2023: Focusing on the project plan and creating the outline
Week 2 starting April 3, 2023: Setting up the basic repository and the whisper models
Week 3 starting April 10, 2023: Creating our NER and sentiment analysis models and the BERT based model
Week 4 starting April 17, 2023: Creating the visualizations, presentation and writing the comparison
**What do you aim to have completed by April 12th (one week before the due date)?**
We hope to have the models implemented by the week before the due date
**Who in your group will be in charge of which component?**
We will be pair programming for the majority of the project as it is just the two of us so we can bounce ideas off each other.

### Repo Structure

The main project is stored in the `final_project.ipynb` Jupyter notebook which pulls from the `ner_lstm.py`, `sent_analysis_bert.py`, `sent_analysis_finetuned.py`, and `sent_analysis_naive_bayes.py` files. Additionally, the project dashboard is stored within `app.py` and the `static` and `templates` folders and can be run by calling `python app.py` or `python3 app.py` depending on your PATH variables. The remaining files are stored models, project data, datasets, or other required files.
