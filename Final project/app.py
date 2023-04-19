import os
from flask import Flask, request, redirect, render_template
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, BertForQuestionAnswering, BertTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import torch
import spacy
from transformers import pipeline
from spacy import displacy
from flaskext.markdown import Markdown

UPLOAD_FOLDER = 'files'

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
nlp = spacy.load("en_core_web_sm")
pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
questionmodel = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
questiontokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Markdown(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    text = ""
    ner = ""
    sentimentText = ""
    sentimentScore = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            os.system(r'ffmpeg -i "audio.webm" -vn -y "audio.wav"')
            input_audio, _ = librosa.load('audio.wav', sr=16000)
            input_feature = processor(input_audio, sampling_rate=16000, return_tensors='pt').input_values
            output_text = model(input_feature).logits.argmax(dim=-1)
            text = processor.decode(output_text[0])

            # use spacy and bert to analyze ner and sentiment
            doc = nlp(text)
            ner = displacy.render(doc, style="ent")
            
            sentiment = pipe(text)
            sentimentText = sentiment[0]['label']
            sentimentScore = sentiment[0]['score']

            # Visualize the input audio waveform
            plt.plot(input_audio)
            plt.title('Input Audio Waveform')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.savefig('static/waveform.png')
            plt.close()

    return render_template('index.html', transcript=text, ner=ner, sentimentText=sentimentText, sentimentScore=sentimentScore)


@app.route('/answer', methods=['GET', 'POST'])
def answer():
    question = ""
    context = ""
    answer = ""
    ner = ""
    sentimentText = ""
    sentimentScore = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            os.system(r'ffmpeg -i "question.webm" -vn -y "question.wav" && ffmpeg -i "context.webm" -vn -y "context.wav"')
            input_audio, _ = librosa.load('question.wav', sr=16000)
            input_feature = processor(input_audio, sampling_rate=16000, return_tensors='pt').input_values
            output_text = model(input_feature).logits.argmax(dim=-1)
            questionText = processor.decode(output_text[0])

            input_audio, _ = librosa.load('context.wav', sr=16000)
            input_feature = processor(input_audio, sampling_rate=16000, return_tensors='pt').input_values
            output_text = model(input_feature).logits.argmax(dim=-1)
            contextText = processor.decode(output_text[0])            

            # Extract input question and context from request
            question = questionText
            context = contextText

            # Tokenize input question and context
            input_dict = questiontokenizer.encode_plus(question, context, return_tensors='pt')

            # Get model output
            outputs = questionmodel(**input_dict)

            # Extract answer from model output
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = questiontokenizer.convert_tokens_to_string(questiontokenizer.convert_ids_to_tokens(input_dict['input_ids'][0][answer_start:answer_end]))

            # same as index page
            # use spacy and bert to analyze ner and sentiment
            doc = nlp(question)
            ner = displacy.render(doc, style="ent")
            
            sentiment = pipe(question)
            sentimentText = sentiment[0]['label']
            sentimentScore = sentiment[0]['score']

            # Visualize the input audio waveform
            plt.plot(input_audio)
            plt.title('Input Audio Waveform')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.savefig('static/waveform.png')
            plt.close()

    return render_template('answer.html', question=question, context=context, answer=answer, ner=ner, sentimentText=sentimentText, sentimentScore=sentimentScore)


if __name__ == '__main__':
    app.run()