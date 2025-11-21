import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Function to analyze email (no caching, models loaded each time)
def analyze_email(email_body):
    """Analyzes an email for spam and sentiment, returning result type and message."""
    spam_pipeline = pipeline("text-classification", model="cybersectony/phishing-email-detection-distilbert_v2.4.1")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("ISOM5240GP4/email_sentiment", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # this is to handle if user does not input anything
    if not email_body.strip():
        return "error", "Email body is empty. Please provide an email to analyze."
    
    try:
        # Step 1: Check if the email is spam
        spam_result = spam_pipeline(email_body)
        spam_label = spam_result[0]["label"]  # LABEL_1 indicates spam
        spam_confidence = spam_result[0]["score"]
        
        if spam_label == "LABEL_1":
            return "spam", f"This is a spam email (Confidence: {spam_confidence:.2f}). No follow-up needed."
        else:
            # Step 2: Analyze sentiment for non-spam emails
            inputs = tokenizer(email_body, padding=True, truncation=True, return_tensors='pt')
            # Pass the tokenized inputs to the sentiment model
            # **inputs unpacks the dictionary of inputs to the model
            outputs = sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().detach().numpy()
            sentiment_index = np.argmax(predictions)
            sentiment_confidence = predictions[0][sentiment_index]
            # Convert numeric prediction to human-readable sentiment
            # If index is 1, it's Positive, otherwise Negative
            sentiment = "Positive" if sentiment_index == 1 else "Negative"
            
            if sentiment == "Positive":
                return "positive", (f"This email is not spam (Confidence: {spam_confidence:.2f}).\n"
                                    f"Sentiment: {sentiment} (Confidence: {sentiment_confidence:.2f}). No follow-up needed.")
            else:
                return "negative", (f"This email is not spam (Confidence: {spam_confidence:.2f}).\n"
                                    f"Sentiment: {sentiment} (Confidence: {sentiment_confidence:.2f}).\n"
                                    "<b>Need to Follow-Up</b>: This email is not spam and has negative sentiment.")
    except Exception as e:
        return "error", f"An error occurred during analysis: {str(e)}"

# Main application function
def main():
    # Set title and objective
    st.title("EmailSentry")
    st.write("Aims to perform analysis on incoming emails and to determine whether there is urgency or higher priority for the company to follow-up.")
    
    # Initialize session state variables
    if "email_body" not in st.session_state:
        st.session_state.email_body = ""
    if "result" not in st.session_state:
        st.session_state.result = ""
    if "result_type" not in st.session_state:
        st.session_state.result_type = ""
    
    # Instructions section
    with st.expander("How to Use", expanded=False):
        st.write("""
        - Type or paste an email into the text box.
        - Alternatively, click one of the sample buttons to load a predefined email.
        - Press 'Analyze Email' to check if it’s spam and analyze its sentiment.
        - Use 'Clear' to reset the input and result.
        """)
    
    # Text area for email input
    email_body = st.text_area("Email", value=st.session_state.email_body, height=200, key="email_input")
    
    # Define sample emails
    sample_spam = """
Subject: Urgent: Verify Your Account Now!
Dear Customer,
We have detected unusual activity on your account. To prevent suspension, please verify your login details immediately by clicking the link below:
[Click Here to Verify](http://totally-legit-site.com/verify)
Failure to verify within 24 hours will result in your account being locked. This is for your security.
Best regards,
The Security Team
    """
    spam_snippet = "Subject: Urgent: Verify Your Account Now! Dear Customer, We have detected unusual activity..."
    
    sample_not_spam_positive = """
Subject: Great Experience with HKTV mall
Dear Sir,
I just received my order and I’m really impressed with the speed of the delivery. Keep up the good work.
Best regards,
Emily
    """
    positive_snippet = "Subject: Great Experience with HKTV mall Dear Sir, I just received my order and I’m really..."
    
    sample_not_spam_negative = """
Subject: Issue with Recent Delivery
Dear Support,
I received my package today, but it was damaged, and two items were missing. This is really frustrating—please let me know how we can resolve this as soon as possible.
Thanks,
Sarah
    """
    negative_snippet = "Subject: Issue with Recent Delivery Dear Support, I received my package today, but..."
    
    # Display sample buttons
    st.subheader("Examples")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(spam_snippet, key="spam_sample"):
            st.session_state.email_body = sample_spam
            st.session_state.result = ""
            st.session_state.result_type = ""
            st.rerun()
    with col2:
        if st.button(positive_snippet, key="positive_sample"):
            st.session_state.email_body = sample_not_spam_positive
            st.session_state.result = ""
            st.session_state.result_type = ""
            st.rerun()
    with col3:
        if st.button(negative_snippet, key="negative_sample"):
            st.session_state.email_body = sample_not_spam_negative
            st.session_state.result = ""
            st.session_state.result_type = ""
            st.rerun()
    
    # Action buttons
    col_analyze, col_clear = st.columns(2)
    with col_analyze:
        if st.button("Analyze Email", key="analyze", type="primary"):
            if email_body:
                with st.spinner("Analyzing email..."):
                    result_type, result = analyze_email(email_body)
                    st.session_state.result = result
                    st.session_state.result_type = result_type
            else:
                st.session_state.result = "Please enter an email body or select a sample to analyze."
                st.session_state.result_type = ""
    
    with col_clear:
        if st.button("Clear", key="clear"):
            st.session_state.email_body = ""
            st.session_state.result = ""
            st.session_state.result_type = ""
            st.rerun()
    
    # Display analysis result
    if st.session_state.result:
        if st.session_state.result_type == "spam":
            st.markdown(f'<div class="spam-result">{st.session_state.result}</div>', unsafe_allow_html=True)
        elif st.session_state.result_type == "positive":
            st.markdown(f'<div class="positive-result">{st.session_state.result}</div>', unsafe_allow_html=True)
        elif st.session_state.result_type == "negative":
            st.markdown(f'<div class="negative-result">{st.session_state.result}</div>', unsafe_allow_html=True)
        else:
            st.write(st.session_state.result)
    
    # Inject custom CSS with updated result block colors
    st.markdown("""
        <style>
        /* Sample buttons (light grey, small) */
        div.stButton > button[kind="secondary"] {
            font-size: 12px;
            padding: 5px 10px;
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #cccccc;
            border-radius: 3px;
        }
        /* Analyze Email button (orange, larger) */
        div.stButton > button[kind="primary"] {
            background-color: #FF5733;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            margin-right: 10px;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #E74C3C;
        }
        /* Clear button (blue) */
        div.stButton > button[kind="secondary"][key="clear"] {
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
        }
        div.stButton > button[kind="secondary"][key="clear"]:hover {
            background-color: #0056b3;
        }
        /* Result boxes: Red for no follow-up, Green for follow-up */
        .spam-result {
            background-color: #ff3333; /* Red for no follow-up */
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #cc0000;
        }
        .positive-result {
            background-color: #ff3333; /* Red for no follow-up */
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #cc0000;
        }
        .negative-result {
            background-color: #33cc33; /* Green for follow-up needed */
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #009900;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()