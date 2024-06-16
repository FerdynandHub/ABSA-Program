import gradio as gr
import spacy
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load spaCy model for POS tagging and sentence segmentation
nlp = spacy.load("en_core_web_sm")

# Load your custom tokenizer and model
model_path = './Book-reviews-ABSA-Fine-tuned-Distilbert-final'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

def extract_aspects(text):
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == 'NOUN']

def split_into_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def predict_sentiment(aspect, sentence):
    aspect_text = f"Sentence: '{sentence}' Aspect: '{aspect}'"
    inputs = tokenizer(aspect_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1)
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_label = labels[probabilities.argmax()]
    sentiment_score = probabilities.max().item()
    return sentiment_label, sentiment_score

def analyze_text(user_input):
    result = []

    # Process the user input
    sentences = split_into_sentences(user_input)
    aspect_sentiments = {}

    # Process each sentence
    for sentence in sentences:
        aspects = extract_aspects(sentence)
        aspect_sentences = []
        for aspect in aspects:
            sentiment_label, sentiment_score = predict_sentiment(aspect, sentence)
            if aspect not in aspect_sentiments:
                aspect_sentiments[aspect] = []
            aspect_sentiments[aspect].append({'Sentiment': sentiment_label, 'Confidence': sentiment_score})
            aspect_sentences.append(f"Aspect: {aspect}\nSentiment: {sentiment_label}, Confidence: {sentiment_score:.2f}")
        if aspect_sentences:
            result.append(f"Sentence: {sentence}\n" + "\n".join(aspect_sentences) + "\n")

    return "\n".join(result)

# Set up the Gradio interface
iface = gr.Interface(
    fn=analyze_text,
    inputs="text",
    outputs="text",
    title="Aspect-Based Sentiment Analysis",
    description="Enter text to analyze the sentiment of different aspects.",
    allow_flagging="never"  # Disable the flag button
)

# Launch the interface
iface.launch()
