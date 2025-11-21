import streamlit as st
import pandas as pd
import os
import torch
import time
import re
import ast
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# 1. Sentiment Analysis Function
@st.cache_data(show_spinner=False)
def perform_sentiment_analysis(df, text_column="text"):
    """
    Perform sentiment analysis on a DataFrame column using Bertweet.
    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        text_column (str): Column name containing the text to analyze.
    Returns:
        pd.DataFrame: DataFrame with added sentiment labels.
    """
    # Combine post title and selftext
    if 'selftext' in df.columns and 'title' in df.columns:
        df['post_text'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
    else:
        df['post_text'] = df['title']

    # Clean and standardize text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    df['post_text'] = df['post_text'].apply(clean_text)

    # Load the finetuned model and tokenizer
    model_name = "henryliiiiii/bertweet-finetuned-reddit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Function to get sentiment scores
    def get_sentiment(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return [sentiment_map[p] for p in torch.argmax(probabilities, dim=-1).tolist()]

    # Apply sentiment analysis to each text entry
    df["sentiment_label"] = df["post_text"].apply(lambda x: get_sentiment([x])[0])
    return df

# 2. Text Generation Pipeline for Summaries
def get_text_generation_pipeline():
    """
    Returns a HuggingFace text generation pipeline using GPT2.
    """
    return pipeline("text-generation", model="gpt2")

def generate_nikon_summary_with_action(nikon_df, avg_score):
    """
    Generate a concise summary of Nikon Z30 sentiment analysis results with actionable suggestions.
    Args:
        nikon_df (pd.DataFrame): DataFrame filtered for Nikon posts.
        avg_score (float): Average sentiment score.
    Returns:
        str: Summary (<=400 chars) with actionable suggestions.
    """
    pos = nikon_df["sentiment_label"].value_counts().get("Positive", 0)
    neu = nikon_df["sentiment_label"].value_counts().get("Neutral", 0)
    neg = nikon_df["sentiment_label"].value_counts().get("Negative", 0)
    total = pos + neu + neg
    if total == 0:
        return "No Nikon Z30 posts available for summary."
    pct_pos = pos / total * 100
    pct_neu = neu / total * 100
    pct_neg = neg / total * 100

    summary = (
        f"Nikon Z30 Reddit posts: {pct_pos:.0f}% positive, {pct_neu:.0f}% neutral, {pct_neg:.0f}% negative. "
        f"Average sentiment score: {avg_score:.2f} (-1=Negative, 0=Neutral, 1=Positive). "
    )
    return summary[:400]

def show_nikon_z30_summary_button(nikon_df, avg_score, key=None):
    """
    Display a button to generate a summary for Nikon Z30 posts.
    """
    if st.button("No time to read? Click to generate summary!", key=key):
        with st.spinner("Generating summary..."):
            summary = generate_nikon_summary_with_action(nikon_df, avg_score)
        st.success(summary)

def generate_brand_comparison_summary(sentiment_stats):
    """
    Compare sentiment analysis results for Nikon, Sony, and Canon, and generate actionable knowledge for Nikon.
    Args:
        sentiment_stats (dict): Dict with keys as brand names and values as dicts of sentiment counts and avg_score.
    Returns:
        str: Actionable summary for Nikon (<=1000 chars).
    """
    brands = ["nikon", "sony", "canon"]
    lines = []
    for brand in brands:
        stats = sentiment_stats.get(brand, {})
        pos = stats.get("Positive", 0)
        neu = stats.get("Neutral", 0)
        neg = stats.get("Negative", 0)
        avg = stats.get("avg_score", 0)
        total = pos + neu + neg
        if total == 0:
            pct_str = "No data"
        else:
            pct_str = f"{pos/total*100:.0f}% positive, {neu/total*100:.0f}% neutral, {neg/total*100:.0f}% negative, avg: {avg:.2f}"
        lines.append(f"{brand.capitalize()}: {pct_str}")

    # Actionable knowledge for Nikon Z30 product manager
    actionable = [
        "1. Highlight video and vlogging strengths in marketing.",
        "2. Address autofocus and low-light concerns in updates or communications.",
        "3. Expand lens options and promote third-party compatibility."
    ]

    summary = (
        "Sentiment comparison:\n" +
        "\n".join(lines) +
        "\n\nActionable Knowledge for Nikon Z30 Product Manager:\n" +
        "\n".join(actionable)
    )
    return summary[:1000]

def show_competitor_summary_button(filtered_df, avg_score):
    """
    Display a button to generate a summary comparing Nikon, Sony, and Canon.
    """
    sentiment_stats = {}
    for brand in ["nikon", "sony", "canon"]:
        brand_df = filtered_df[filtered_df["camera"].str.lower() == brand]
        counts = brand_df["sentiment_label"].value_counts().to_dict()
        avg = brand_df["sentiment_label"].map({"Negative": -1, "Neutral": 0, "Positive": 1}).mean()
        sentiment_stats[brand] = {
            "Positive": counts.get("Positive", 0),
            "Neutral": counts.get("Neutral", 0),
            "Negative": counts.get("Negative", 0),
            "avg_score": avg if pd.notnull(avg) else 0
        }
    if st.button("No time to read? Click to generate summary!"):
        with st.spinner("Generating summary..."):
            summary = generate_brand_comparison_summary(sentiment_stats)
        st.success(summary)

# 3. Sidebar Navigation
def sidebar():
    """
    Display the sidebar with logo, navigation, and contact info.
    """
    st.sidebar.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/f/f3/Nikon_Logo.svg" width="120"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        """
        <h2 style="text-align:center;">Nikon Z30 Insight Hub</h2>
        """,
        unsafe_allow_html=True
    )
    page = st.sidebar.selectbox(
        label="navigation",
        options=("Nikon Z30 Insights", "Competitor Analysis"),
        index=0, 
        label_visibility="collapsed"
    )
    st.sidebar.markdown('<hr style="border:1px solid #e6e6e6;">', unsafe_allow_html=True)
    st.sidebar.caption('Wish to connect?')
    st.sidebar.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; gap: 12px;">
            <a href="https://www.linkedin.com/in/daisy-huang-5a37351a6/" target="_blank" title="Daisy Huang">
                <img src="https://media.licdn.com/dms/image/v2/D4D35AQGC3-Em8xyEIw/profile-framedphoto-shrink_400_400/profile-framedphoto-shrink_400_400/0/1700539170740?e=1748246400&v=beta&t=FYGP5Tlp9LmTtP0vUYajVXE74nf9YU6clcsc-2DqpAI" width="60" style="border-radius:50%;" />
            </a>
            <a href="https://www.linkedin.com/in/henry-li-52051b2a5/" target="_blank" title="Henry Li">
                <img src="https://media.licdn.com/dms/image/v2/D5635AQEWIrquMmsx5Q/profile-framedphoto-shrink_400_400/profile-framedphoto-shrink_400_400/0/1710083972342?e=1748250000&v=beta&t=RiOK-bFOtXak8Bq-qaPAcBG4n9kE59YmTCWbPLvC9Vk" width="60" style="border-radius:50%;" />
            </a>
        </div>
        <div style="text-align:center;"></div>
        """,
        unsafe_allow_html=True
    )
    return page

# 4. Nikon Z30 Insights Page
def show_nikon_z30():
    """
    Display the main insights page for Nikon Z30, including sentiment metrics and summary generator.
    """
    st.header("🎉 Welcome to Nikon Z30 Insight Hub!")
    st.markdown(
        """
        <div style="background: #fffbe6; padding: 20px 24px; border-radius: 12px; margin-bottom: 24px; border: 1px solid transparent;">
            <span style="font-size: 1.1rem; color: #222;">
                This is a strategic social media intelligence platform designed for Nikon to drive market success through AI-powered insights. Through tracking user comments on the key social media platform, Reddit, the hub can help Nikon:<br><br>
                - 📊 Identify user satisfaction rate and pain points effortlessly<br>
                - 🔍 Facilitates direct benchmarking of the Z30 against key competitors (e.g., Canon R50, Sony ZVE10)<br>
                - 💪 Generates actionable summaries of reviews and ratings to inform quick decisions on product improvements, marketing, or warranties.<br><br>
                <b>Simplify Insight, Amplify Impact ✨</b>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    # Load the prepared CSV file
    try:
        df = pd.read_csv("cleaned_all_reddit_posts.csv")
        if "created_utc" in df.columns:
            df["created_date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
        else:
            df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    except Exception as e:
        st.error(f"Error reading the prepared file: {e}")
        return

    # Sentiment analysis with progress bar
    with st.spinner("Performing sentiment analysis, please wait..."):
        progress_bar = st.progress(0, text="Analyzing sentiment...")
        total = len(df)
        batch_size = 32

        def batch_sentiment(df, batch_size=32):
            results = []
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size]
                batch_result = perform_sentiment_analysis(batch, text_column="text")
                results.append(batch_result)
                progress_bar.progress(min((i+batch_size)/total, 1.0), text=f"Analyzing sentiment... ({min(i+batch_size, total)}/{total})")
            progress_bar.empty()
            return pd.concat(results, ignore_index=True)

        df = batch_sentiment(df, batch_size=batch_size)

    # Filter for Nikon-related posts
    nikon_df = df[df["camera"].astype(str).str.strip().str.lower() == "nikon"].copy()

    # Map sentiment_label to numeric score
    sentiment_score_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
    nikon_df["sentiment_score"] = nikon_df["sentiment_label"].map(sentiment_score_map)
    avg_score = nikon_df["sentiment_score"].mean()

    # Show summary generator button before metrics
    show_nikon_z30_summary_button(nikon_df, avg_score, key="summary_before_metrics")

    # Metric 1: Sentiment distribution pie chart
    st.subheader("📊 Sentiment Distribution for Z30-related Posts")
    nikon_sentiment_counts = nikon_df["sentiment_label"].value_counts()
    fig = px.pie(
        nikon_sentiment_counts,
        names=nikon_sentiment_counts.index,
        values=nikon_sentiment_counts.values,
        color=nikon_sentiment_counts.index,
        color_discrete_map={
            "Negative": "#b8b5cb",
            "Neutral": "#fff3a3",
            "Positive": "#ffd700"
        }
    )
    st.plotly_chart(fig)

    # Metric 2: Average sentiment score for Nikon posts
    st.subheader("⭐ Average Sentiment Score for Z30-related Posts")
    st.metric("Average Sentiment Score", f"{avg_score:.2f}", help="-1: Negative, 0: Neutral, 1: Positive")
    import plotly.graph_objects as go
    fig_score = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_score if pd.notnull(avg_score) else 0,
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "#ffe066"},
            'steps': [
                {'range': [-1, 0], 'color': "#fffbe6"},
                {'range': [0, 1], 'color': "#fffde6"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': avg_score if pd.notnull(avg_score) else 0
            }
        },
        title={'text': "Nikon Posts Sentiment Score"}
    ))
    st.plotly_chart(fig_score)

    # Metric 3: Top posts for Z30 by upvotes
    st.subheader("🔥 Top Nikon Posts by Upvotes")
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce")
    nikon_posts = df[df["camera"].str.lower() == "nikon"].sort_values("upvotes", ascending=False).head(20)
    st.dataframe(nikon_posts[["upvotes", "sentiment_label", "title", "selftext", "created_date", "num_comments", "post_link"]])

    # Metric 4: Top posts for Z30 by number of comments
    st.subheader("💬 Top Nikon Posts by Number of Comments")
    df["num_comments"] = pd.to_numeric(df["num_comments"], errors="coerce")
    nikon_discussion_posts = df[df["camera"].str.lower() == "nikon"].sort_values("num_comments", ascending=False).head(20)
    st.dataframe(nikon_discussion_posts[["num_comments", "upvotes", "sentiment_label", "title", "selftext", "created_date", "post_link"]])

    # Metric 5: User Activity Heatmap (Post Count Over Time)
    st.subheader("📈 User Activity Heatmap (Post Count Over Time)")
    nikon_df = df[df["camera"].astype(str).str.strip().str.lower() == "nikon"].copy()
    nikon_df = nikon_df[nikon_df["created_date"].notna()]
    nikon_df["period"] = nikon_df["created_date"].dt.to_period("Q")
    min_period = pd.Period("2022Q3")
    nikon_df = nikon_df[nikon_df["period"] >= min_period]
    post_count = nikon_df.groupby("period").size().rename("Post Count").reset_index()
    if not post_count.empty:
        all_periods = pd.period_range(start="2022Q3", end=nikon_df["period"].max(), freq="Q")
        post_count = post_count.set_index("period").reindex(all_periods, fill_value=0).reset_index()
        post_count.rename(columns={"index": "period"}, inplace=True)
    heatmap_pivot = post_count.pivot_table(columns="period", values="Post Count", aggfunc="sum").fillna(0)
    heatmap_pivot.columns = heatmap_pivot.columns.astype(str)
    fig, ax = plt.subplots(figsize=(max(4, len(heatmap_pivot.columns) * 0.8), 2))
    sns.heatmap(
        heatmap_pivot,
        annot=True,
        fmt=".0f",
        cmap=sns.color_palette(["#fffbe6", "#ffe066", "#ffd700"], as_cmap=True),
        cbar_kws={'label': 'Number of Posts'},
        ax=ax
    )
    ax.set_xlabel("Post Created Date")
    ax.set_title("User Activity Heatmap (Nikon Post Count Over Time)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)

# 5. Competitor Analysis Page
def load_data():
    """
    Load and preprocess the main Reddit posts dataset.
    """
    df = pd.read_csv("cleaned_all_reddit_posts.csv")
    if "created_utc" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    else:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["text"] = df["title"].fillna('') + " " + df["selftext"].fillna('')
    df["text"] = df["text"].fillna("")
    if "sentiment_label" not in df.columns or df["sentiment_label"].isnull().all() or (df["sentiment_label"] == "").all():
        df = perform_sentiment_analysis(df, text_column="text")
    return df

def show_competitor_analysis():
    """
    Display the competitor analysis page, including filters, tables, heatmaps, and wordclouds.
    """
    df = load_data()
    st.markdown(
        """
        <div style="background: #fffbe6; padding: 20px 24px; border-radius: 12px; margin-bottom: 24px; border: 1px solid transparent;">
            <div style="font-size: 1.08rem; color: #222; margin-bottom: 10px;">
                In the dynamic world of photography, three cameras dominate novice creators’ radar:
            </div>
            <div style="text-align:center; font-size:1.18rem; font-weight:bold; color:#222; margin-bottom: 10px;">
                <b>📷 Nikon Z30, Sony ZVE10, and Canon R50</b>
            </div>
            <div style="font-size: 1.08rem; color: #222; margin-bottom: 10px;">
                As the "big three" brands’ entry-level anchors, the Sony ZVE10 and Canon R50 are more than rivals to Nikon—they are pivotal benchmarks shaping market expectations as gateways to visual storytelling for beginners.
            </div>
            <div style="font-size: 1.08rem; color: #222;">
                Stay tuned to uncover why these models demand our focus in crafting the Z30’s competitive edge! 💡
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Show Parameter Comparison table
    try:
        param_df = pd.read_csv("Parameter Comparison.csv")
        st.subheader("📋 Camera Parameter Comparison")
        st.markdown(
            """
            <style>
            .centered-table {margin-left:auto; margin-right:auto;}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(param_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to load Parameter Comparison table: {e}")

    left_col, right_col = st.columns([1, 3])

    # Filters for competitor-related posts
    with left_col:
        st.header("🔍 Filters")
        camera_options = ["nikon", "sony", "canon"]
        selected_cameras = st.multiselect(
            "Select Cameras",
            camera_options,
            default=camera_options
        )
        start_date = st.date_input("Start Date", df["created_date"].min().date())
        end_date = st.date_input("End Date", df["created_date"].max().date())
        filtered = df[
            (df["camera"].isin(selected_cameras)) &
            (df["created_date"].dt.date >= start_date) &
            (df["created_date"].dt.date <= end_date)
        ].copy()
    
    # Top posts by upvotes and sentiment heatmap
    with right_col:
        st.subheader("🔥 Top Posts by Upvotes (Filtered)")
        display_cols = [col for col in ["camera", "upvotes", "sentiment_label", "title", "selftext", "created_date", "num_comments"] if col in filtered.columns]
        top_posts = filtered.sort_values("upvotes", ascending=False).head(10)
        st.dataframe(top_posts[display_cols])

        st.subheader("📌 Sentiment Heatmap by Camera Model")
        if "sentiment_label" in filtered.columns:
            sentiment_order = ["Negative", "Neutral", "Positive"]
            heat_df = filtered.pivot_table(
                index="camera",
                columns="sentiment_label",
                aggfunc="size",
                fill_value=0
            )
            heat_df = heat_df.reindex(columns=sentiment_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(heat_df, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
            ax.set_xlabel("Sentiment Label")
            ax.set_ylabel("Camera Model")
            st.pyplot(fig)
        else:
            st.info("No sentiment_label column available for heatmap.")

        # Feature wordclouds
        st.subheader("🧠 Feature WordClouds")
        feature_keywords = {
            "Image Quality": ["image", "quality", "photo", "sharp", "noise", "dynamic range"],
            "Autofocus": ["autofocus", "focus", "tracking", "eye af", "subject detect"],
            "Video": ["video", "recording", "4k", "fps", "vlog"],
            "Portability": ["light", "compact", "small", "carry", "travel"],
            "Buttons": ["button", "layout", "control", "touch", "menu"],
            "Battery": ["battery", "life", "charge", "duration", "power"]
        }

        def generate_wordcloud(text):
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            return fig

        for feature, keywords in feature_keywords.items():
            relevant_texts = filtered["text"].apply(
                lambda x: any(kw in x.lower() for kw in keywords) if isinstance(x, str) else False
            )
            combined_text = " ".join(filtered[relevant_texts]["text"].dropna().tolist())
            if combined_text.strip():
                st.markdown(f"**{feature}**")
                fig_wc = generate_wordcloud(combined_text)
                st.pyplot(fig_wc)

# 6. main function
if __name__ == "__main__":
    st.set_page_config(page_title="Nikon Z30 Insight Hub", layout="wide")
    page = sidebar()
    if page == "Nikon Z30 Insights":
        show_nikon_z30()
    elif page == "Competitor Analysis":
        show_competitor_analysis()