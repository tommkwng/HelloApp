import streamlit as st
from transformers import pipeline


# ================== Model Loading ==================
@st.cache_resource
def load_summarization_pipeline():
    return pipeline("summarization", model="wyiyiyiyi/results_med_dialog", tokenizer="facebook/bart-large-cnn")

@st.cache_resource
def load_diagnosis_pipeline():
    return pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

@st.cache_resource
def load_zero_shot_pipeline():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ================== UI Configuration ==================
urgency_emojis = {
    "Highly Urgent": "🚑",
    "Urgent": "⚠️",
    "Moderate": "ℹ️",
    "Non-Urgent": "😊"
}

st.set_page_config(page_title="Medical Assistant", layout="wide")

# ================== Collapsible Sidebar ==================
with st.sidebar:
    st.title("🏥 MedAssist Pro")
    with st.expander("ℹ️ Platform Guide", expanded=False):
        st.markdown("""
        **Processing Workflow**:
        1. **Symptom Extraction**  
           - Automatically identifies key symptoms
        2. **Preliminary Diagnosis**  
           - Generates initial medical assessment
        3. **Urgency Evaluation**  
           - Classifies case priority level
        **Recommended Input**:  
        > "I've had persistent chest pain and dizziness since morning"
        """)
    
    st.markdown("---")
    st.caption("v2.1 | [Report Issues](https://example.com)")

# ================== Main Interface ==================
st.title("Intelligent Medical Assistant")
user_input = st.text_area(
    "Describe your symptoms:",
    value="Hi, my sister delivered a baby boy just then and was told that the pulse rate is high and the blood sugar is low. is under some kind of observation and not given to his mother. The reason they give is bcoz my sister was having sugar during pregnancy. Is it a serious problem? Will the baby be just fine. Please reply. Doctor: in pregnant mother due to diabetes sugar will be high so baby will get more sugar so it will be producing more insulin.insulin is the prime hormone for growth in infants.{yes,not growth hormone:it s only after 5 years}up to 5 years hormone:thyroxineso more insulin produced more chubby baby is.after deliveryMORE INSULIN:LESS SUGAR{BCOZ BABY IS OUT}so blood sugar lowit s not serious if baby is monitered regulated sugar regularlybut main effect will be the baby is going to be bit heavier side and most probably may get diabetes milletus in future.so advice is keep him on low glycemic index foods after 7 years",
    height=300
)

if st.button("Start Analysis", type="primary"):
    if not user_input.strip():
        st.error("Please enter valid symptoms")
    else:
        with st.spinner("Analyzing symptoms..."):
            # ===== Step 1: Symptom Summary =====
            with st.container():
                st.subheader("📋 Symptom Summary")
                try:
                    summarizer = load_summarization_pipeline()
                    summary = summarizer(
                        f"Extract key symptoms: {user_input}",
                        max_length=80,
                        min_length=20
                    )[0]['summary_text']
                    st.info(f"**Identified Symptoms**: {summary}")
                except Exception as e:
                    st.error(f"Summary Error: {str(e)}")

            # ===== Step 2: Preliminary Diagnosis =====
            with st.container():
                st.subheader("🩺 Preliminary Diagnosis")
                try:
                    diagnoser = load_diagnosis_pipeline()
                    
                    # Qwen-specific chat format
                    messages = [
                        {"role": "system", "content": "You are a medical expert. Analyze symptoms and provide structured diagnosis."},
                        {"role": "user", "content": f"""
                         Analyze these symptoms: {summary}
                         
                         Required response format:
                         1. Differential diagnosis (3 possibilities)
                         2. Most likely condition
                         3. Recommended next steps
                         """}
                    ]
                    
                    # Generate response
                    diagnosis = diagnoser(
                        messages,
                        max_new_tokens=500,  # Qwen需要更多token
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        eos_token_id=151645,  # Qwen的特定结束符<|endoftext|>
                        return_full_text=False
                    )[0]['generated_text']
                    
                    # 结构化解析响应
                    formatted_diagnosis = "\n".join([
                        line for line in diagnosis.split("\n") 
                        if line.startswith(("1.", "2.", "3."))
                    ])
                    
                    st.success(f"""
                    **Clinical Assessment**:
                    {formatted_diagnosis}
                    """)
                    
                except Exception as e:
                    st.error(f"Diagnosis Error: {str(e)}")

            # ===== Step 3: Urgency Classification =====
            with st.container():
                st.subheader("🚨 Urgency Evaluation")
                try:
                    classifier = load_zero_shot_pipeline()
                    classification = classifier(
                        summary,
                        candidate_labels=list(urgency_emojis.keys())
                    )
                    
                    cols = st.columns(4)
                    for idx, (label, score) in enumerate(zip(classification["labels"], classification["scores"])):
                        with cols[idx]:
                            st.markdown(f"""
                            <div style="text-align:center">
                                <h3>{urgency_emojis[label]}</h3>
                                <strong>{label}</strong><br>
                                {score*100:.1f}%
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Classification Error: {str(e)}")