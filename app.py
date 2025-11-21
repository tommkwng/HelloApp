import gradio as gr
import torch
import csv, os, datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
#root {
    max-width: 900px;
    margin: 40px auto;
    padding: 20px;
    background: var(--block-background);
    border-radius: 8px;
    box-shadow: var(--block-container-elevation-2);
}
#sidebar {
    background: var(--block-container-background);
    padding: 16px;
    border-radius: 8px;
}
#primary-btn { background-color: #2a9d8f !important; color: white !important; }
.gradio-txt, .gradio-btn { margin-bottom: 12px !important; }
"""

# ─── Model Loading ──────────────────────────────────────────────────────────────
def load_models():
    clf_tok   = AutoTokenizer.from_pretrained("prachijhaveri/roberta-fake-news-detector")
    clf_model = AutoModelForSequenceClassification.from_pretrained(
        "prachijhaveri/roberta-fake-news-detector"
    )
    gen_tok   = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    if torch.cuda.is_available():
        clf_model.cuda()
        gen_model.cuda()
    return clf_tok, clf_model, gen_tok, gen_model

# load once
clf_tok, clf_model, gen_tok, gen_model = load_models()


# ─── Confidence Bar Helper ─────────────────────────────────────────────────────
def confidence_bar(is_fake: bool, pct: float) -> str:
    color = "#e63946" if is_fake else "#2a9d8f"
    return (
        "<div style='width:100%;background:var(--block-border);border-radius:4px;"
        "margin:6px 0;'>"
        f"<div style='width:{pct:.1f}%;background:{color};padding:4px 0;"
        "border-radius:4px;text-align:center;color:white;font-weight:bold;"
        "font-size:14px;'>"
        f"{pct:.1f}% confidence</div></div>"
    )


# ─── Core Inference Routine ────────────────────────────────────────────────────
def analyze(
    headline: str,
    date: str,
    subject: str,
    speaker: str,
    speaker_descr: str,
    state_info: str,
    context_src: str,
    full_article: str,
    user_justification: str,
    clf_tok,
    clf_model,
    gen_tok,
    gen_model,
):
    text = full_article.strip()
    if not text:
        return "<p style='color:#e63946'>❗ Please paste the full article text.</p>", "", "", ""
    snippet = text[:2000]

    # 1) Build metadata blob
    metadata_blob = (
        f"Headline: {headline}\n"
        f"Date: {date}\n"
        f"Subject: {subject}\n"
        f"Speaker: {speaker}\n"
        f"Speaker description: {speaker_descr}\n"
        f"State: {state_info}\n"
        f"Context: {context_src}"
    )

    # 2) Justification (echo or generate)
    if user_justification.strip():
        just_text = user_justification.strip()
    else:
        just_prompt = (
            f"{metadata_blob}\n\nArticle (snippet):\n{snippet}\n\n"
            "Briefly justify why this article might be fake or real."
        )
        j_tok = gen_tok(
            just_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,   # summariser can also use 512 tokens
        )
        if torch.cuda.is_available():
            j_tok = {k:v.cuda() for k,v in j_tok.items()}
        with torch.no_grad():
            j_ids = gen_model.generate(
                **j_tok,
                max_new_tokens=60,
                num_beams=1,
                do_sample=False,
                early_stopping=True,
            )
        just_text = gen_tok.decode(j_ids[0], skip_special_tokens=True).strip()

    # 3) Classification on enriched text, total max 512 tokens
    enriched_blob = metadata_blob + f"\nJustification: {just_text}"
    inp = clf_tok(
        enriched_blob,
        snippet,
        return_tensors="pt",
        truncation="only_second",   # keep enriched_blob intact
        padding="max_length",
        max_length=512,             # MUST be ≤ 512 for RoBERTa
    )
    if torch.cuda.is_available():
        inp = {k:v.cuda() for k,v in inp.items()}
    with torch.no_grad():
        logits = clf_model(**inp).logits
    probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    is_fake = bool(probs.argmax())
    label   = "Fake" if is_fake else "Real"
    conf    = probs.max()*100
    verdict_html = (
        f"<h3 style='margin:4px 0;color:{'#e63946' if is_fake else '#2a9d8f'}'>{label}</h3>"
        + confidence_bar(is_fake, conf)
    )

    # 4) Summarisation on full text (unchanged)
    sum_prompt = (
        f"The article is classified as '{label}' ({conf:.1f}% confidence). "
        "Summarize it in 2–3 concise sentences:\n\n" + text
    )
    sum_tok = gen_tok(
        sum_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=2048,
    )
    if torch.cuda.is_available():
        sum_tok = {k:v.cuda() for k,v in sum_tok.items()}
    with torch.no_grad():
        sum_ids = gen_model.generate(
            **sum_tok,
            max_new_tokens=80,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
        )
    summary_html = f"<p style='margin:4px 0;font-size:14px;'>{gen_tok.decode(sum_ids[0], skip_special_tokens=True).strip()}</p>"

    # 5) Justification output + feedback log
    just_html = f"<p style='margin:4px 0;font-size:14px;'><strong>Justification:</strong> {just_text}</p>"
    cls_input = enriched_blob + "\n\nArticle:\n" + snippet
    return verdict_html, summary_html, just_html, cls_input


# ─── Feedback Logger ────────────────────────────────────────────────────────────
def log_feedback(cls_input: str, feedback: str):
    ts = datetime.datetime.utcnow().isoformat()
    fname = "feedback_log.csv"
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "classifier_input", "feedback"])
        w.writerow([ts, cls_input.replace("\n", " | "), feedback])
    return "<p style='color:#2a9d8f'>Thank you for your feedback!</p>"


# ─── UI ─────────────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("## 📰 Fake-News Detector & Summariser")

        with gr.Row():
            with gr.Column(scale=1, elem_id="sidebar"):
                gr.Markdown("### Metadata")
                headline_tb      = gr.Textbox(label="Article Headline *", lines=2)
                date_tb          = gr.Textbox(label="Date (YYYY-MM-DD)")
                subject_tb       = gr.Textbox(label="Subject")
                state_tb         = gr.Textbox(label="State info")
                speaker_tb       = gr.Textbox(label="Speaker/Writer")
                speaker_desc_tb  = gr.Textbox(label="Speaker description")
                context_tb       = gr.Textbox(label="Context Source", lines=2)
                justification_tb = gr.Textbox(label="Your Justification (optional)", lines=3)

            with gr.Column(scale=2):
                full_article_tb    = gr.Textbox(label="Full Article *", lines=20)
                analyze_btn        = gr.Button("Analyze & Summarise", variant="primary", elem_id="primary-btn")
                verdict_html       = gr.HTML("<p>Waiting for analysis…</p>")
                summary_html       = gr.HTML("<p>Waiting for summary…</p>")
                justification_html = gr.HTML("<p>Waiting for justification…</p>")
                record_box         = gr.Textbox(visible=False)

        analyze_btn.click(
            fn=lambda *args: analyze(*args, clf_tok, clf_model, gen_tok, gen_model),
            inputs=[
                headline_tb, date_tb, subject_tb, speaker_tb,
                speaker_desc_tb, state_tb, context_tb,
                full_article_tb, justification_tb
            ],
            outputs=[verdict_html, summary_html, justification_html, record_box],
            show_progress=True,
        )

        feedback_radio = gr.Radio(choices=["Yes", "No"], label="Was the verdict correct?")
        feedback_btn   = gr.Button("Submit Feedback")
        feedback_msg   = gr.HTML()

        feedback_btn.click(
            fn=log_feedback,
            inputs=[record_box, feedback_radio],
            outputs=[feedback_msg],
        )

    return demo


def main():
    ui = build_ui()
    ui.launch()


if __name__ == "__main__":
    main()