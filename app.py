import os
import nest_asyncio
nest_asyncio.apply()
import streamlit as st
from transformers import pipeline
from huggingface_hub import login
from streamlit.components.v1 import html
import pandas as pd
import torch
import random
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Retrieve the token from environment variables for Hugging Face login
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    st.stop()

# Hugging Face login with the token just in case the models requirement authorization
login(token=hf_token)

# Timer component using HTML and JavaScript
def timer():
    return """
    <div id="timer" style="font-size:16px;color:#666;margin-bottom:10px;">⏱️ Elapsed: 00:00</div>
    <script>
    (function() {
        var start = Date.now();
        var timerElement = document.getElementById('timer');
        localStorage.removeItem("freezeTimer");
        var interval = setInterval(function() {
            if(localStorage.getItem("freezeTimer") === "true"){
                clearInterval(interval);
                timerElement.style.color = '#00cc00';
                return;
            }
            var elapsed = Date.now() - start;
            var minutes = Math.floor(elapsed / 60000);
            var seconds = Math.floor((elapsed % 60000) / 1000);
            timerElement.innerHTML = '⏱️ Elapsed: ' +
            (minutes < 10 ? '0' : '') + minutes + ':' +
            (seconds < 10 ? '0' : '') + seconds;
        }, 1000);
    })();
    </script>
    """
# Display the Title
st.set_page_config(page_title="Twitter/X Tweets Scorer & Report Generator", page_icon="📝")
st.header("𝕏/Twitter Tweets Sentiment Report Generator")

# Concise introduction
st.write("This model🎰 will score your tweets in your CSV file🗄️ based on their sentiment😀 and generate a report🗟 answering your query question❔ based on those results.")

# Display VRAM status for debug
def print_gpu_status(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        st.info(f"{label}: Allocated {allocated:.2f} GB, Reserved {reserved:.2f} GB")

# Cache the model loading functions
@st.cache_resource
def get_sentiment_model():
    return pipeline("text-classification", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                    device=0 if torch.cuda.is_available() else -1)

@st.cache_resource
def get_summary_model():
    return pipeline("text-generation", 
                   model="frankai98/T5FinetunedCommentSummary",
                   device=0 if torch.cuda.is_available() else -1)

# Function to clear GPU memory
def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
# Function to build the prompt for text generation model using only the sentiment label
def build_prompt(query_input, sampled_docs):
    docs_text = ""
    # Use the sentiment label directly from each document (converted to lowercase)
    for idx, doc in enumerate(sampled_docs):
        sentiment_word = doc['sentiment'].lower() if doc.get('sentiment') else "unknown"
        docs_text += f"Tweet {idx+1} (Sentiment: {sentiment_word}): {doc['comment']}\n"

    system_message = """You are an helpful assistant. Read the Tweets with their sentiment (Negative, Neutral, Positive) provided and produce a well-structured report that answers the query question.
Your task:
- Summarize both positive and negative aspects, highlighting any trends in user sentiment.
- Include an introduction, key insights, and a conclusion, reaching about 400 words.
- DO NOT repeat these instructions or the user's query in the final report. Only provide the final text."""

    user_content = f"""**Tweets**:
{docs_text}

**Query Question**: "{query_input}"

Now produce the final report only, without reiterating these instructions or the query."""
    # This is the required chat format for Llama-3.2-1B-Instruct model, select intruct model for better performance
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_content}]
            }
        ]
    ]
    return messages

# Main Function Part:
def main():
    # Let the user specify the column name for tweets text (defaulting to "content")
    tweets_column = st.text_input("Enter the column name for Tweets🐦:", value="content")
    
    # Input: Query question for analysis and CSV file upload for candidate tweets
    query_input = st.text_area("Enter your query question❓for analysis (Format: How do these people feel about ...?) (this does not need to be part of the CSV):")
    uploaded_file = st.file_uploader(f"Upload Tweets CSV File < 1MB🗄️(must contain a '{tweets_column}' column with preferably <1000 tweets)", type=["csv"])
    # Error check steps to ensure that the uploaded file meets the requirements
    candidate_docs = []
    if uploaded_file is not None:
        if uploaded_file.size > 1 * 1024 * 1024:
            st.error("The file is too large! Please upload a file smaller than 1MB.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if tweets_column not in df.columns:
                    st.error(f"CSV must contain a '{tweets_column}' column.")
                else:
                    candidate_docs = df[tweets_column].dropna().astype(str).tolist()
                    st.write("File uploaded successfully!🎆")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
    # Click on the button will start running the pipelines in sequence and the timer
    if st.button("Generate Report"):
        st.session_state.setdefault("timer_started", False)
        st.session_state.setdefault("timer_frozen", False)
        if uploaded_file is None:
            st.error("Please upload a CSV file🗄️.")
        elif not tweets_column.strip():
            st.error("Please enter your column name")
        elif not candidate_docs:
            st.error(f"CSV must contain a '{tweets_column}' column.")
        elif not query_input.strip():
            st.error("Please enter a query question❔!")
        else:
            if not st.session_state.timer_started and not st.session_state.timer_frozen:
                st.session_state.timer_started = True
                html(timer(), height=50)
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            processed_docs = []
            scored_results = []
            
            # Check which documents need summarization (tweets longer than 280 characters)
            docs_to_summarize = []
            docs_indices = []
            for i, doc in enumerate(candidate_docs):
                if len(doc) > 280:
                    docs_to_summarize.append(doc)
                    docs_indices.append(i)
            
            # Summarize long tweets if needed
            if docs_to_summarize:
                status_text.markdown("**📝 Loading summarization model...**")
                t5_pipe = get_summary_model()
                status_text.markdown("**📝 Summarizing long tweets...**")
                # Dispay the progress
                for idx, (i, doc) in enumerate(zip(docs_indices, docs_to_summarize)):
                    progress = int((idx / len(docs_to_summarize)) * 25)
                    progress_bar.progress(progress)
                    input_text = "summarize: " + doc
                    try:
                        summary_result = t5_pipe(
                            input_text, 
                            max_length=128,
                            min_length=10,
                            no_repeat_ngram_size=2,
                            num_beams=4,
                            early_stopping=True,
                            truncation=True
                        )
                        candidate_docs[i] = summary_result[0]['generated_text']
                    except Exception as e:
                        st.warning(f"Error summarizing document {i}: {str(e)}")
                # Delete summarization model from VRAM for optimization
                del t5_pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Load sentiment analysis model
            status_text.markdown("**🔍 Loading sentiment analysis model...**")
            progress_bar.progress(25)
            score_pipe = get_sentiment_model()
            status_text.markdown("**🔍 Scoring documents...**")
            
            for i, doc in enumerate(candidate_docs):
                progress_offset = 25 if docs_to_summarize else 0
                progress = progress_offset + int((i / len(candidate_docs)) * (50 - progress_offset))
                progress_bar.progress(progress)
                try:
                    result = score_pipe(doc, truncation=True, max_length=512)
                    if isinstance(result, list):
                        result = result[0]
                    processed_docs.append(doc)
                    # Store only the sentiment label (e.g., "Negative", "Neutral", "Positive")
                    scored_results.append(result)
                except Exception as e:
                    st.warning(f"Error scoring document {i}: {str(e)}")
                    processed_docs.append("Error processing this document")
                    scored_results.append({"label": "Neutral"})
                
                if i % max(1, len(candidate_docs) // 10) == 0:
                    status_text.markdown(f"**🔍 Scoring documents... ({i}/{len(candidate_docs)})**")
            
            # Pair documents with sentiment labels using key "comment"
            scored_docs = [
                {"comment": doc, "sentiment": result.get("label", "Neutral")}
                for doc, result in zip(processed_docs, scored_results)
            ]
            # Delete sentiment analysis model from VRAM for optimization
            del score_pipe
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            status_text.markdown("**📊 Loading report generation model...**")
            progress_bar.progress(67)
            # Clear the VRAM to prepare for Llama so it wouldn't encounter OOM errors
            clear_gpu_memory()
            
            status_text.markdown("**📝 Preparing data for report generation...**")
            progress_bar.progress(75)
            # Set the maximum examples text generation model can take
            max_tweets = 1000
            if len(scored_docs) > max_tweets:
                sampled_docs = random.sample(scored_docs, max_tweets)
                st.info(f"Sampling {max_tweets} out of {len(scored_docs)} tweets for report generation")
            else:
                sampled_docs = scored_docs
            
            prompt = build_prompt(query_input, sampled_docs)
            # Define the text generation pipeline
            def process_with_llama(prompt):
                try:
                    pipe = pipeline(
                        "text-generation",
                        model="unsloth/Llama-3.2-1B-Instruct",
                        device="cuda" if torch.cuda.is_available() else -1,
                        torch_dtype=torch.bfloat16,
                    )
                    result = pipe(prompt, max_new_tokens=400, return_full_text=False)
                    return result, None
                except Exception as e:
                    return None, str(e)

            status_text.markdown("**📝 Generating report with Llama...**")
            progress_bar.progress(80)
            
            raw_result, error = process_with_llama(prompt)
            # Process the result to get the report or display the error
            if error:
                st.error(f"Gemma processing failed: {str(error)}")
                report = "Error generating report. Please try again with fewer tweets."
            else:
                report = raw_result[0][0]['generated_text']
            # Clear the VRAM in the end so it won't affect the next app run    
            clear_gpu_memory()
            progress_bar.progress(100)
            status_text.success("**✅ Generation complete!**")
            html("<script>localStorage.setItem('freezeTimer', 'true');</script>", height=0)
            st.session_state.timer_frozen = True
            # Replace special characters for the correct format
            formatted_report = report.replace('\n', '<br>')
            
            st.subheader("Generated Report:")
            st.markdown(f"<div style='font-size: normal; font-weight: normal;'>{formatted_report}</div>", unsafe_allow_html=True)
# Running the main function            
if __name__ == '__main__':
    main()
