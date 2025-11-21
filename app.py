##########################################
# Step 0: Import required libraries
##########################################
import streamlit as st  # For web interface
from transformers import (
    pipeline,  # For loading pre-trained models
    SpeechT5Processor,  # For text-to-speech processing
    SpeechT5ForTextToSpeech,  # TTS model
    SpeechT5HifiGan,  # Vocoder for generating audio waveforms
    AutoModelForCausalLM,  # For text generation
    AutoTokenizer  # For tokenizing input text
)  # AI model components

from datasets import load_dataset  # To load voice embeddings
import torch  # For tensor computations
import soundfile as sf  # For handling audio files
import re  # For regular expressions in text processing

##########################################
# Initial configuration
##########################################
st.set_page_config(
    page_title="Just Comment",  # Title of the web app
    page_icon="💬",  # Icon displayed in the browser tab
    layout="centered",  # Center the layout of the app
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

##########################################
# Global model loading with caching
##########################################
@st.cache_resource(show_spinner=False)  # Cache the models for performance
def _load_models():
    """Load and cache all ML models with optimized settings"""
    return {
        # Emotion classification pipeline
        'emotion': pipeline(
            "text-classification",  # Specify task type
            model="Thea231/jhartmann_emotion_finetuning",  # Load the model
            truncation=True  # Enable text truncation for long inputs
        ),
        
        # Text generation components
        'textgen_tokenizer': AutoTokenizer.from_pretrained(
            "Qwen/Qwen1.5-0.5B",  # Load tokenizer
            use_fast=True  # Enable fast tokenization
        ),
        'textgen_model': AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen1.5-0.5B",  # Load text generation model
            torch_dtype=torch.float16  # Use half-precision for faster inference
        ),
        
        # Text-to-speech components
        'tts_processor': SpeechT5Processor.from_pretrained("microsoft/speecht5_tts"),  # Load TTS processor
        'tts_model': SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts"),  # Load TTS model
        'tts_vocoder': SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan"),  # Load vocoder
        
        # Preloaded speaker embeddings
        'speaker_embeddings': torch.tensor(
            load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")[7306]["xvector"]  # Load speaker embeddings
        ).unsqueeze(0)  # Add an additional dimension for batch processing
    }

##########################################
# UI Components
##########################################
def _display_interface():
    """Render user interface elements"""
    st.title("Just Comment")  # Set the main title of the app
    st.markdown("### I'm listening to you, my friend～")  # Subheading for user interaction
    
    return st.text_area(
        "📝 Enter your comment:",  # Label for the text area
        placeholder="Type your message here...",  # Placeholder text
        height=150,  # Height of the text area
        key="user_input"  # Unique key for the text area
    )

##########################################
# Core Processing Functions
##########################################
def _analyze_emotion(text, classifier):
    """Identify dominant emotion with confidence threshold"""
    results = classifier(text, return_all_scores=True)[0]  # Get emotion scores
    valid_emotions = {'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'}  # Define valid emotions
    filtered = [e for e in results if e['label'].lower() in valid_emotions]  # Filter results by valid emotions
    return max(filtered, key=lambda x: x['score'])  # Return the emotion with the highest score

def _generate_prompt(text, emotion):
    """Create structured prompts for all emotion types"""
    prompt_templates = {
        "sadness": (
            "Sadness detected: {input}\n"
            "Required response structure:\n"
            "1. Empathetic acknowledgment\n2. Support offer\n3. Solution proposal\n"
            "Response:"
        ),
        "joy": (
            "Joy detected: {input}\n"
            "Required response structure:\n"
            "1. Enthusiastic thanks\n2. Positive reinforcement\n3. Future engagement\n"
            "Response:"
        ),
        "love": (
            "Affection detected: {input}\n"
            "Required response structure:\n"
            "1. Warm appreciation\n2. Community focus\n3. Exclusive benefit\n"
            "Response:"
        ),
        "anger": (
            "Anger detected: {input}\n"
            "Required response structure:\n"
            "1. Sincere apology\n2. Action steps\n3. Compensation\n"
            "Response:"
        ),
        "fear": (
            "Concern detected: {input}\n"
            "Required response structure:\n"
            "1. Reassurance\n2. Safety measures\n3. Support options\n"
            "Response:"
        ),
        "surprise": (
            "Surprise detected: {input}\n"
            "Required response structure:\n"
            "1. Acknowledge uniqueness\n2. Creative solution\n3. Follow-up\n"
            "Response:"
        )
    }
    return prompt_templates.get(emotion.lower(), "").format(input=text)  # Format and return the appropriate prompt

def _process_response(raw_text):
    """Clean and format the generated response"""
    # Extract text after last "Response:" marker
    processed = raw_text.split("Response:")[-1].strip()
    
    # Remove incomplete sentences
    if '.' in processed:
        processed = processed.rsplit('.', 1)[0] + '.'  # Ensure the response ends with a period
    
    # Ensure length between 50-200 characters
    return processed[:200].strip() if len(processed) > 50 else "Thank you for your feedback. We value your input and will respond shortly."

def _generate_text_response(input_text, models):
    """Generate optimized text response with timing controls"""
    # Emotion analysis
    emotion = _analyze_emotion(input_text, models['emotion'])  # Analyze the emotion of user input
    
    # Prompt engineering
    prompt = _generate_prompt(input_text, emotion['label'])  # Generate prompt based on detected emotion
    
    # Text generation with optimized parameters
    inputs = models['textgen_tokenizer'](prompt, return_tensors="pt").to('cpu')  # Tokenize the prompt
    outputs = models['textgen_model'].generate(
        inputs.input_ids,  # Input token IDs
        max_new_tokens=100,  # Strict token limit for response length
        temperature=0.7,  # Control randomness in text generation
        top_p=0.9,  # Control diversity in sampling
        do_sample=True,  # Enable sampling to generate varied responses
        pad_token_id=models['textgen_tokenizer'].eos_token_id  # Use end-of-sequence token for padding
    )
    
    return _process_response(
        models['textgen_tokenizer'].decode(outputs[0], skip_special_tokens=True)  # Decode and process the response
    )

def _generate_audio_response(text, models):
    """Convert text to speech with performance optimizations"""
    # Process text input for TTS
    inputs = models['tts_processor'](text=text, return_tensors="pt")  # Tokenize input text for TTS
    
    # Generate spectrogram
    spectrogram = models['tts_model'].generate_speech(
        inputs["input_ids"],  # Input token IDs for TTS
        models['speaker_embeddings']  # Use preloaded speaker embeddings
    )
    
    # Generate waveform with optimizations
    with torch.no_grad():  # Disable gradient calculation for inference
        waveform = models['tts_vocoder'](spectrogram)  # Generate audio waveform from spectrogram
    
    # Save audio file
    sf.write("response.wav", waveform.numpy(), samplerate=16000)  # Save waveform as a WAV file
    return "response.wav"  # Return the path to the saved audio file

##########################################
# Main Application Flow
##########################################
def main():
    """Primary execution flow"""
    # Load models once
    ml_models = _load_models()  # Load all models and cache them
    
    # Display interface
    user_input = _display_interface()  # Show the user input interface
    
    if user_input:  # Check if user has entered input
        # Text generation stage
        with st.spinner("🔍 Analyzing emotions and generating response..."):  # Show loading spinner
            text_response = _generate_text_response(user_input, ml_models)  # Generate text response
        
        # Display results
        st.subheader("📄 Generated Response")  # Subheader for response section
        st.markdown(f"```\n{text_response}\n```")  # Display generated response in markdown format
        
        # Audio generation stage
        with st.spinner("🔊 Converting to speech..."):  # Show loading spinner
            audio_file = _generate_audio_response(text_response, ml_models)  # Generate audio response
            st.audio(audio_file, format="audio/wav")  # Play audio file in the app

if __name__ == "__main__":
    main()  # Execute the main function when the script is run