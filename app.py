import gradio as gr
import json
import os
import torch
import nltk
import spacy
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Download necessary NLTK data for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

# Global loading of models and NLP components
fin_model = None
summarizer = None
ner_model = None
auth_token = os.environ.get("HF_Token")  # For NER model loading

def load_models():
    global fin_model, summarizer, ner_model
    
    # Load sentiment analysis model
    print("Loading sentiment model...")
    try:
        fin_model = pipeline("sentiment-analysis", model="ylingag/ISOM5240_financial_tone")
        print("Sentiment model loaded successfully.")
    except Exception as e:
        print(f"Failed to load sentiment model: {e}")
        fin_model = None
    
    # Load summarization model
    print("Loading summarization model...")
    try:
        summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
        print("Summarization model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load summarization model: {e}")
        print("Will continue without summarization capability.")
        summarizer = None
    
    # Load NER model directly using pipeline
    print("Loading NER model...")
    try:
        ner_model = pipeline("ner", model="dslim/bert-base-NER")
        print("NER model loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load NER model: {e}")
        print("Will continue without NER capability.")
        ner_model = None

def split_in_sentences(text):
    """Split text into sentences"""
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

def make_spans(text, results):
    """Create highlighted text spans with sentiment labels"""
    results_list = []
    for i in range(len(results)):
        # Ensure we display specific sentiment labels, not LABEL format
        label = results[i]['label']
        # If the label is in LABEL_ format, replace with specific sentiment terms
        if label.startswith("LABEL_"):
            if label == "LABEL_0":
                label = "Negative"
            elif label == "LABEL_1":
                label = "Neutral"
            elif label == "LABEL_2":
                label = "Positive"
        results_list.append(label)
    spans = list(zip(split_in_sentences(text), results_list))
    return spans

def text_to_sentiment(text):
    """Analyze overall sentiment of the text"""
    global fin_model
    if not fin_model:
        return "Sentiment model not available."
    
    if not text or not text.strip():
        return "Please enter text for analysis."
    
    try:
        sentiment = fin_model(text)[0]["label"]
        # If the label is in LABEL_ format, replace with specific sentiment terms
        if sentiment.startswith("LABEL_"):
            if sentiment == "LABEL_0":
                sentiment = "Negative"
            elif sentiment == "LABEL_1":
                sentiment = "Neutral"
            elif sentiment == "LABEL_2":
                sentiment = "Positive"
        return sentiment
    except Exception as e:
        print(f"Error during overall sentiment analysis: {e}")
        return f"Error: {str(e)}"

def summarize_text(text):
    """Generate a summary for longer text"""
    global summarizer
    if not summarizer:
        return "Summarization model not available."
    
    if not text or len(text.strip()) < 50:
        return "Text too short for summarization."
    
    try:
        resp = summarizer(text)
        return resp[0]['summary_text']
    except Exception as e:
        print(f"Error during summarization: {e}")
        return f"Summarization error: {str(e)}"

def fin_ext(text):
    """Analyze sentiment of each sentence in the text for highlighting"""
    global fin_model
    if not fin_model or not text:
        return None
    
    try:
        results = fin_model(split_in_sentences(text))
        return make_spans(text, results)
    except Exception as e:
        print(f"Error during sentence-level sentiment analysis: {e}")
        return None

def identify_entities(text):
    """Identify entities using NER model and spaCy as backup"""
    global ner_model
    if not text:
        return None
    
    try:
        # First, try to use the transformer-based NER model
        if ner_model:
            entities = ner_model(text)
            
            # Process NER results into spans format for HighlightedText
            spans = []
            last_end = 0
            current_position = 0
            
            # Sort entities by their position
            sorted_entities = sorted(entities, key=lambda x: x['start'])
            
            for entity in sorted_entities:
                # Get entity position and label
                start = entity['start']
                end = entity['end']
                entity_text = entity['word']
                entity_type = entity['entity']
                
                # Add text before entity
                if start > last_end:
                    spans.append((text[last_end:start], None))
                
                # Add the entity with its type
                spans.append((entity_text, entity_type))
                last_end = end
            
            # Add remaining text
            if last_end < len(text):
                spans.append((text[last_end:], None))
            
            return spans
        
        # If transformer model failed, fallback to spaCy
        else:
            doc = nlp(text)
            spans = []
            last_end = 0
            
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC", "ORG"]:  # Only locations and organizations
                    start = text.find(ent.text, last_end)
                    if start != -1:
                        end = start + len(ent.text)
                        if start > last_end:
                            spans.append((text[last_end:start], None))
                        spans.append((ent.text, ent.label_))
                        last_end = end
            
            if last_end < len(text):
                spans.append((text[last_end:], None))
            
            return spans
            
    except Exception as e:
        print(f"Error during entity identification: {e}")
        # Fallback to spaCy if error occurred
        try:
            doc = nlp(text)
            spans = []
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC", "ORG"]:
                    spans.append((ent.text, ent.label_))
            
            # If no entities found, return special message
            if not spans:
                spans = [(text, None)]
            
            return spans
        except:
            # Last resort
            return [(text, None)]

def analyze_financial_text(text):
    """Master function that performs all analysis tasks"""
    if not text or not text.strip():
        return None, "No summary available.", None, "No sentiment available."
    
    # Generate summary
    summary = summarize_text(text)
    
    # Perform overall sentiment analysis
    overall_sentiment = text_to_sentiment(text)
    
    # Perform sentence-level sentiment analysis with highlighting
    sentiment_spans = fin_ext(text)
    
    # Identify entities with highlighting
    entity_spans = identify_entities(text)
    
    return sentiment_spans, summary, entity_spans, overall_sentiment

# Try to load models at app startup
try:
    load_models()
except Exception as e:
    print(f"Initial model loading failed: {e}")
    # Gradio interface will still start, but functionality will be limited

# Gradio interface definition
app_title = "Financial Tone Analysis"
app_description = "The project will summarize financial news content, analyze financial sentiment, and flag relevant companies and countries"

with gr.Blocks(title=app_title) as iface:
    gr.Markdown(f"# {app_title}")
    gr.Markdown(app_description)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=10, 
                label="Financial News Text", 
                placeholder="Enter a longer financial news text here for analysis...",
                value="US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month."
            )
            analyze_btn = gr.Button("Start Analysis", variant="primary")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Text Summary")
            summary_output = gr.Textbox(label="Summary", lines=3)
            
    with gr.Row():
        gr.Markdown("### Market sentiment")
        with gr.Column(scale=1):
            gr.Markdown("#### Overall Tone")
            overall_sentiment_output = gr.Label(label="Document Sentiment")
        with gr.Column(scale=2):
            gr.Markdown("#### Sentence-by-Sentence Analysis")
            sentiment_output = gr.HighlightedText(label="Financial Tone by Sentence")
            
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Interested Parties")
            entities_output = gr.HighlightedText(label="Identified Companies & Locations")
    
    # Set up the click event for the analyze button
    analyze_btn.click(
        fn=analyze_financial_text, 
        inputs=[input_text], 
        outputs=[sentiment_output, summary_output, entities_output, overall_sentiment_output]
    )

if __name__ == "__main__":
    print("Starting Gradio application...")
    # share=True will generate a public link
    iface.launch(share=True) 