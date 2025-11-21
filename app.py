import os
os.environ["HOME"] = os.getcwd()  # 解决streamlit权限问题，确保Streamlit能在当前目录下创建配置文件

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import base64

# --------------------
# 0. Page config & sidebar
# --------------------
st.set_page_config(page_title='Weibo Sentiment Analysis & Auto Report', page_icon='💡', layout='wide')
st.sidebar.image('https://huggingface.co/front/assets/huggingface_logo-noborder.svg', width=120)
st.sidebar.markdown('''
**Weibo Sentiment Analysis & Auto Report System**  
- Automatic sentiment classification
- Sentiment distribution visualization
- Auto-generated analysis report
- Downloadable results
''')

# --------------------
# 1. Load sentiment analysis model (huggingface云端模型)
# --------------------
@st.cache_resource
def load_sentiment_model():
    # 这里填写你在huggingface上模型的名字
    model_dir = 'Erica12345612/weibo-sentiment-bert'
    # 自动下载并加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # 创建情感分析pipeline，自动选择GPU或CPU
    pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    return pipe

sentiment_pipe = load_sentiment_model()

# --------------------
# 2. Load English report generation model (gpt2, huggingface云端模型)
# --------------------
@st.cache_resource
def load_report_model():
    # 直接从huggingface云端加载gpt2模型
    model_dir = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model

gen_tokenizer, gen_model = load_report_model()

def generate_summary_keyword(statistics):
    prompt = (
        f"Summarize the overall user sentiment for Weibo in one short English sentence, based on these results: {statistics}"
    )
    input_ids = gen_tokenizer(prompt, return_tensors='pt').input_ids
    output = gen_model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=gen_tokenizer.eos_token_id,
        no_repeat_ngram_size=3
    )
    keyword = gen_tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return keyword

def generate_report(statistics):
    keyword = generate_summary_keyword(statistics)
    stat_dict = {}
    for item in statistics.split(','):
        if ':' in item:
            k, v = item.split(':')
            stat_dict[k.strip()] = v.strip()
    if stat_dict:
        main_sentiment = max(stat_dict, key=lambda k: int(stat_dict[k]))
        main_count = stat_dict[main_sentiment]
    else:
        main_sentiment = 'N/A'
        main_count = '0'
    summary = f"The overall sentiment among Weibo users is {keyword}."
    report = f'''
Sentiment Analysis Report

Summary:
{summary}
The sentiment analysis of recent Weibo posts shows the following distribution: {statistics}.
Among all posts, the most common sentiment is "{main_sentiment}" with {main_count} occurrences.

Possible Reasons:
A high proportion of 'none' sentiment may indicate that users are posting more neutral or informational content, or that the sentiment detection model needs further tuning for the Weibo context. This distribution may also be influenced by recent events, product updates, or public opinion trends. Positive sentiments such as 'like' and 'happiness' indicate user satisfaction, while 'none' or negative sentiments may reflect dissatisfaction or lack of engagement.

Business Implications:
Weibo can use these insights to optimize content recommendation and user engagement strategies. The company should leverage positive feedback to reinforce strengths, while paying close attention to negative or neutral sentiments to identify areas for improvement. Understanding the root causes behind these sentiments can help guide business strategy and improve platform experience.

Suggestions for Improvement:
1. Regularly monitor sentiment trends to detect changes in user attitudes.
2. Engage with users who express negative or neutral sentiments to gather feedback.
3. Promote positive experiences and address common pain points.
4. Use sentiment insights to inform product and service enhancements.

This report is generated automatically by the Weibo Sentiment Analysis & Auto Report System. It can be used for product optimization, user operations, and strategic decision support.
'''
    return report.strip()

# --------------------
# 3. Download sample CSV
# --------------------
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="sample_weibo.csv">Download sample CSV</a>'
    return href

sample_df = pd.DataFrame({
    'text': [
        '今天心情特别好，阳光真美！',
        '这个服务太差了，真让人生气。',
        '有点失落，事情没按预期发展。',
        '收到惊喜礼物，好开心！',
        '没什么特别的感觉，就是普通一天。',
        '最近压力很大，有点害怕未来。',
        '我很喜欢这款产品，推荐！',
        '看到这些消息真的很恶心。',
        '客服态度非常好，点赞。',
        '快递太慢了，失望。',
        '绝对爱上了这个功能！',
        '和预期不一样，有点失望。',
        '新版本更新很棒，体验提升了。',
        '为什么总是卡顿？',
        '售后支持很及时，满意。',
        '感觉被忽视了，有点难过。',
        '这是我用过最棒的应用。',
        '再也不会买这家东西了。',
        '操作很方便，省心省力。',
        '太糟糕了，体验极差。'
    ]
})

# --------------------
# 4. Streamlit UI
# --------------------
st.markdown('<h1 style="color:#FF6F00;font-size:2.5em;">Weibo Sentiment Analysis & Auto Report</h1>', unsafe_allow_html=True)
st.markdown('<hr style="border:1px solid #FF6F00;">', unsafe_allow_html=True)
st.write('**Upload or input Weibo texts. The system will analyze sentiment distribution, generate visualizations, and auto-generate a brief report.**')

st.markdown(get_table_download_link(sample_df), unsafe_allow_html=True)
st.info("Please upload a CSV file with a column named 'text'. You can download a sample above.")

texts = []
input_mode = st.radio('Select input method:', ['Batch upload CSV', 'Manual input'])

if input_mode == 'Batch upload CSV':
    uploaded_file = st.file_uploader('Upload a CSV file with a column named "text"', type=['csv'])
    st.write("uploaded_file:", uploaded_file)
    if uploaded_file is not None:
        st.success(f"File {uploaded_file.name} uploaded successfully!")
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size)
        try:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.write("Columns:", df.columns.tolist())
        except Exception as e:
            st.error(f"Error reading file: {e}")
        if 'text' not in df.columns:
            st.error('CSV file must contain a column named "text"!')
            st.stop()
        texts = df['text'].astype(str).tolist()
        st.write("Texts loaded:", texts[:3])
else:
    for i in range(5):
        text = st.text_input(f'Input Weibo text #{i+1} (optional)')
        if text:
            texts.append(text)

if texts:
    st.markdown('---')
    st.subheader('1. 🎯 Sentiment Analysis Results')
    with st.spinner('Analyzing sentiment...'):
        results = sentiment_pipe(texts)
    label_map = {
        '0': 'like', '1': 'disgust', '2': 'happiness', '3': 'sadness',
        '4': 'anger', '5': 'surprise', '6': 'fear', '7': 'none',
        'like': 'like', 'disgust': 'disgust', 'happiness': 'happiness',
        'sadness': 'sadness', 'anger': 'anger', 'surprise': 'surprise',
        'fear': 'fear', 'none': 'none',
        'LABEL_0': 'like', 'LABEL_1': 'disgust', 'LABEL_2': 'happiness', 'LABEL_3': 'sadness',
        'LABEL_4': 'anger', 'LABEL_5': 'surprise', 'LABEL_6': 'fear', 'LABEL_7': 'none'
    }
    pred_labels = [label_map.get(str(r['label']), r['label']) for r in results]
    df_result = pd.DataFrame({'Text': texts, 'Sentiment': pred_labels, 'Confidence': [round(r['score'], 3) for r in results]})
    st.dataframe(df_result.style.background_gradient(cmap='Oranges'))

    st.markdown('---')
    st.subheader('2. 📊 Sentiment Distribution Visualization')
    stat = df_result['Sentiment'].value_counts().sort_index()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.Paired.colors
    stat.plot.pie(autopct='%1.1f%%', ax=ax[0], title='Sentiment Distribution (Pie)', colors=colors)
    stat.plot.bar(ax=ax[1], title='Sentiment Distribution (Bar)', color='#FF6F00')
    ax[0].set_ylabel('')
    st.pyplot(fig)

    st.markdown('---')
    st.subheader('3. 📝 Auto-generated Analysis Report')
    stat_str = ', '.join([f'{k}: {v}' for k, v in stat.items()])
    with st.spinner('Generating report...'):
        report = generate_report(stat_str)
    st.success(report)

    st.markdown('---')
    st.subheader('4. ⬇️ Download Results')
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button('Download Results CSV', csv, 'sentiment_results.csv', 'text/csv')
else:
    st.info('Please upload a CSV file or manually input Weibo texts.')