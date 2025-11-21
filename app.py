# app.py
import streamlit as st
from transformers import pipeline
import time

st.set_page_config(
    page_title="Cosmetic Review Analyst",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.session_state.disable_watchdog = True

def load_css():
    st.markdown("""
    <style>
        .reportview-container .main .block-container{
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextInput textarea {
            border-radius: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #ff6b6b, #ff8e53);
        }
        .st-bw {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models():
    summarizer = pipeline(
        "summarization",
        model="Falconsai/text_summarization",
        max_length=200,
        temperature=0.7
    )
    
    classifier = pipeline(
        "text-classification",
        model="clb5114/EPR_emoclass_TinyBERT",
        return_all_scores=True
    )
    return summarizer, classifier

def main():
    load_css()
    st.title("💄 Cosmetic Review AI Analyst")
    st.warning("⚠️ Please keep reviews under 200 words for optimal analysis")
    
    user_input = st.text_area(
        "Input cosmetic product review (Chinese/English supported)", 
        height=200,
        placeholder="Example: This serum transformed my skin in just 3 days...",
        help="Maximum 200 characters recommended"
    )
    
    if st.button("Start Analysis", use_container_width=True):
        if not user_input.strip():
            st.error("⚠️ Please input valid review content")
            return
            
        with st.spinner('🔍 Analyzing...'):
            try:
                summarizer, classifier = load_models()
                
                with st.expander("Original Review", expanded=True):
                    st.write(user_input)
                
                # Text summarization
                summary = summarizer(user_input, max_length=200)[0]['summary_text']
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.subheader("📝 Summary")
                    with col2:
                        st.markdown(f"```\n{summary}\n```")
                
                # Sentiment analysis
                results = classifier(summary)
                positive_score = results[0][1]['score']
                label = "Positive 👍" if positive_score > 0.5 else "Negative 👎"
                
                with st.container():
                    st.subheader("📊 Sentiment Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Verdict", label)
                        st.write(f"Confidence: {positive_score:.2%}")
                    with col2:
                        progress_color = "#4CAF50" if label=="Positive 👍" else "#FF5252"
                        st.markdown(f"""
                        <div style="
                            background: {progress_color}10;
                            border-radius: 10px;
                            padding: 15px;
                        ">
                            <div style="font-size: 14px; color: {progress_color}; margin-bottom: 8px;">Intensity</div>
                            <div style="height: 8px; background: #eee; border-radius: 4px;">
                                <div style="width: {positive_score*100}%; height: 100%; background: {progress_color}; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()