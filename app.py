import gradio as gr
from transformers import pipeline
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for Gradio)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import time
import io
from PIL import Image

# ── Load model once at startup ────────────────────────────────────────────────
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"

print("⏳ Loading sentiment model... (first run downloads ~260 MB)")
classifier = pipeline("text-classification", model=MODEL_ID)
print("✅ Model loaded successfully!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_sentiment(text: str):
    """Classify a single piece of text and return styled HTML + detail string."""
    if not text or not text.strip():
        return (
            "<div class='empty-msg'>⬆️ Enter some text above to analyse sentiment.</div>",
            "",
        )

    start  = time.time()
    result = classifier(text.strip(), truncation=True, max_length=512)[0]
    elapsed = (time.time() - start) * 1000

    label = result["label"]
    score = result["score"]
    pct   = score * 100

    if label == "POSITIVE":
        emoji, color, bg, bar_color = "😊", "#22c55e", "#dcfce7", "#16a34a"
    else:
        emoji, color, bg, bar_color = "😔", "#ef4444", "#fee2e2", "#b91c1c"

    label_html = f"""
    <div style="background:{bg};border:2px solid {color};border-radius:16px;
                padding:24px 32px;text-align:center;font-family:'Segoe UI',sans-serif;">
        <div style="font-size:52px;margin-bottom:8px;">{emoji}</div>
        <div style="font-size:28px;font-weight:700;color:{color};letter-spacing:2px;">{label}</div>
        <div style="font-size:15px;color:#555;margin-top:6px;">
            Confidence: <strong>{pct:.1f}%</strong>
        </div>
        <div style="background:#e5e7eb;border-radius:999px;height:10px;margin-top:14px;overflow:hidden;">
            <div style="background:{bar_color};width:{pct:.1f}%;height:100%;
                        border-radius:999px;transition:width 0.4s ease;"></div>
        </div>
    </div>"""

    details = (
        f"Model  : {MODEL_ID}\n"
        f"Label  : {label}\n"
        f"Score  : {score:.6f}\n"
        f"Latency: {elapsed:.1f} ms"
    )
    return label_html, details


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Bulk Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def build_pie_chart(pos_count: int, neg_count: int) -> Image.Image:
    """Render and return a matplotlib pie chart as a PIL Image."""
    total = pos_count + neg_count
    sizes  = [pos_count, neg_count]
    colors = ["#22c55e", "#ef4444"]
    labels = [
        f"Positive\n{pos_count} ({pos_count/total*100:.1f}%)",
        f"Negative\n{neg_count} ({neg_count/total*100:.1f}%)",
    ]

    fig, ax = plt.subplots(figsize=(5.5, 4.5), facecolor="#f8fafc")
    ax.set_facecolor("#f8fafc")

    wedges, texts = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3),   # donut
        shadow=False,
    )

    # Center label
    ax.text(0, 0, f"{total}\ntweets", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#334155")

    # Legend
    legend_patches = [
        mpatches.Patch(color=colors[i], label=labels[i]) for i in range(2)
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=11)

    ax.set_title("Sentiment Distribution", fontsize=15, fontweight="bold",
                 color="#1e293b", pad=14)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_summary_html(results: list[dict]) -> str:
    """Return an HTML table listing every tweet and its result."""
    pos = [r for r in results if r["label"] == "POSITIVE"]
    neg = [r for r in results if r["label"] == "NEGATIVE"]
    total = len(results)

    rows = ""
    for i, r in enumerate(results, 1):
        lbl   = r["label"]
        score = r["score"]
        tweet = r["tweet"][:80] + ("…" if len(r["tweet"]) > 80 else "")
        color = "#16a34a" if lbl == "POSITIVE" else "#b91c1c"
        bg    = "#f0fdf4" if lbl == "POSITIVE" else "#fef2f2"
        emoji = "😊" if lbl == "POSITIVE" else "😔"
        rows += f"""
        <tr style="background:{bg};">
            <td style="padding:8px 10px;color:#64748b;font-size:13px;">{i}</td>
            <td style="padding:8px 10px;font-size:13px;color:#1e293b;">{tweet}</td>
            <td style="padding:8px 10px;font-weight:700;color:{color};font-size:13px;">
                {emoji} {lbl}
            </td>
            <td style="padding:8px 10px;color:#475569;font-size:13px;">{score*100:.1f}%</td>
        </tr>"""

    summary_bar = f"""
    <div style="display:flex;gap:12px;margin-bottom:16px;font-family:'Segoe UI',sans-serif;">
        <div style="flex:1;background:#dcfce7;border:1.5px solid #22c55e;border-radius:12px;
                    padding:14px;text-align:center;">
            <div style="font-size:26px;font-weight:800;color:#16a34a;">{len(pos)}</div>
            <div style="font-size:13px;color:#15803d;margin-top:2px;">😊 Positive</div>
        </div>
        <div style="flex:1;background:#fee2e2;border:1.5px solid #ef4444;border-radius:12px;
                    padding:14px;text-align:center;">
            <div style="font-size:26px;font-weight:800;color:#b91c1c;">{len(neg)}</div>
            <div style="font-size:13px;color:#991b1b;margin-top:2px;">😔 Negative</div>
        </div>
        <div style="flex:1;background:#eff6ff;border:1.5px solid #3b82f6;border-radius:12px;
                    padding:14px;text-align:center;">
            <div style="font-size:26px;font-weight:800;color:#1d4ed8;">{total}</div>
            <div style="font-size:13px;color:#1e40af;margin-top:2px;">📊 Total</div>
        </div>
    </div>"""

    table = f"""
    <div style="font-family:'Segoe UI',sans-serif;">
        {summary_bar}
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;border-radius:10px;overflow:hidden;
                      box-shadow:0 1px 6px rgba(0,0,0,0.08);">
            <thead>
                <tr style="background:#1d9bf0;color:white;">
                    <th style="padding:10px;text-align:left;font-size:13px;">#</th>
                    <th style="padding:10px;text-align:left;font-size:13px;">Tweet</th>
                    <th style="padding:10px;text-align:left;font-size:13px;">Sentiment</th>
                    <th style="padding:10px;text-align:left;font-size:13px;">Confidence</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>"""
    return table


def analyze_bulk(raw_text: str):
    """
    Loop through pasted tweets (one per line), classify each,
    and return (pie_chart_image, summary_html).
    """
    # Guard: empty input
    if not raw_text or not raw_text.strip():
        return (
            None,
            "<div class='empty-msg'>⬆️ Paste at least one tweet above and click Analyse.</div>",
        )

    tweets = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

    if len(tweets) == 0:
        return (
            None,
            "<div class='empty-msg'>No valid tweets found. Make sure each tweet is on its own line.</div>",
        )

    # Classify every tweet
    results = []
    for tweet in tweets:
        raw = classifier(tweet, truncation=True, max_length=512)[0]
        results.append({"tweet": tweet, "label": raw["label"], "score": raw["score"]})

    pos_count = sum(1 for r in results if r["label"] == "POSITIVE")
    neg_count = len(results) - pos_count

    chart_img    = build_pie_chart(pos_count, neg_count)
    summary_html = build_summary_html(results)

    return chart_img, summary_html


# ═══════════════════════════════════════════════════════════════════════════════
# Example tweets for Bulk tab
# ═══════════════════════════════════════════════════════════════════════════════
BULK_EXAMPLE = """I absolutely love this new update! It's perfect 🚀
Worst experience ever — completely broken and useless.
Just had the best coffee at that new place downtown ☕
Why does nothing ever work properly?! So frustrated.
Feeling grateful for all the support from this community 💙
This is honestly the most disappointing product I've bought.
Amazing customer service, they solved my issue in minutes!
Can't believe how bad this turned out. Total waste of money.
Had such a wonderful time at the event today 🎉
The app keeps crashing, absolutely terrible."""


SINGLE_EXAMPLES = [
    ["I absolutely love this new feature! It's a game changer 🚀"],
    ["This is the worst service I have ever experienced. Completely disappointed."],
    ["Just tried the new coffee shop downtown — highly recommend it! ☕"],
    ["Why is everything so broken?! I can't believe this keeps happening."],
    ["Feeling grateful for all the support from this amazing community 💙"],
]


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
body, .gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
#header {
    background: linear-gradient(135deg, #1d9bf0 0%, #1a6ec7 100%);
    border-radius: 20px;
    padding: 36px 32px 28px;
    text-align: center;
    color: white;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(29,155,240,0.30);
}
#header h1  { font-size:2rem; font-weight:800; margin:0 0 6px; letter-spacing:-0.5px; }
#header p   { font-size:1rem; opacity:0.88; margin:0; }

#input-box textarea, #bulk-input textarea {
    font-size:15px !important; border-radius:12px !important;
    border:2px solid #e2e8f0 !important; padding:14px !important;
    transition:border-color 0.2s; resize:vertical;
}
#input-box textarea:focus, #bulk-input textarea:focus {
    border-color:#1d9bf0 !important;
    box-shadow:0 0 0 3px rgba(29,155,240,0.15) !important;
}
#analyze-btn, #bulk-btn {
    background: linear-gradient(135deg,#1d9bf0,#1a6ec7) !important;
    border:none !important; border-radius:12px !important;
    color:white !important; font-size:16px !important; font-weight:600 !important;
    padding:12px 0 !important; cursor:pointer !important;
    transition:opacity 0.2s,transform 0.1s !important;
    box-shadow:0 4px 14px rgba(29,155,240,0.35) !important;
}
#analyze-btn:hover, #bulk-btn:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }
#clear-btn, #bulk-clear-btn { border-radius:12px !important; font-size:15px !important; font-weight:600 !important; }
#result-html .prose { padding:0 !important; }
.empty-msg { text-align:center; color:#94a3b8; font-size:15px; padding:32px 0; }
#details-box textarea {
    font-family:'Courier New',monospace !important; font-size:13px !important;
    border-radius:10px !important; color:#475569 !important;
}
.gr-examples .gr-button { border-radius:8px !important; font-size:13px !important; }
#footer  { text-align:center; color:#94a3b8; font-size:12px; margin-top:16px; }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Twitter Sentiment Analyser") as demo:

    # ── Shared header ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
        <h1>🐦 Twitter Sentiment Analyser</h1>
        <p>Powered by DistilBERT fine-tuned on SST-2 &nbsp;·&nbsp; Hugging Face Transformers</p>
    </div>
    """)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Single Analysis ────────────────────────────────────────────
        with gr.Tab("🔍 Single Analysis"):
            input_text = gr.Textbox(
                label="✏️  Enter a tweet or any text",
                placeholder="e.g.  I love how this product works, absolutely fantastic! 🎉",
                lines=4, max_lines=8, elem_id="input-box",
            )
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyse Sentiment", variant="primary",  elem_id="analyze-btn")
                clear_btn   = gr.Button("🗑️  Clear",            variant="secondary", elem_id="clear-btn")

            result_html = gr.HTML(
                value="<div class='empty-msg'>⬆️ Enter some text above to analyse sentiment.</div>",
                label="Sentiment Result", elem_id="result-html",
            )
            details_box = gr.Textbox(
                label="📊 Model Details", interactive=False,
                lines=4, elem_id="details-box",
            )
            gr.Examples(
                examples=SINGLE_EXAMPLES, inputs=input_text,
                label="💡 Try an example", examples_per_page=5,
            )

            # Wiring
            analyze_btn.click(fn=analyze_sentiment, inputs=input_text,
                              outputs=[result_html, details_box])
            input_text.submit(fn=analyze_sentiment, inputs=input_text,
                              outputs=[result_html, details_box])
            clear_btn.click(
                fn=lambda: ("",
                            "<div class='empty-msg'>⬆️ Enter some text above to analyse sentiment.</div>",
                            ""),
                inputs=None, outputs=[input_text, result_html, details_box],
            )

        # ── Tab 2: Bulk Analysis ──────────────────────────────────────────────
        with gr.Tab("📊 Bulk Analysis"):
            gr.HTML("""
            <div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:12px;
                        padding:14px 18px;margin-bottom:12px;font-size:14px;color:#1e40af;">
                📋 <strong>Instructions:</strong> Paste one tweet per line in the box below,
                then click <em>Analyse All Tweets</em>. The app will classify each tweet and
                show you a sentiment distribution pie chart plus a detailed results table.
            </div>
            """)

            bulk_input = gr.Textbox(
                label="📝 Paste tweets (one per line)",
                placeholder="I love this!\nThis is terrible.\nAmazing experience!",
                lines=10, max_lines=20, elem_id="bulk-input",
            )
            with gr.Row():
                bulk_btn       = gr.Button("📊 Analyse All Tweets", variant="primary",  elem_id="bulk-btn")
                bulk_clear_btn = gr.Button("🗑️  Clear",             variant="secondary", elem_id="bulk-clear-btn")

            gr.Examples(
                examples=[[BULK_EXAMPLE]],
                inputs=bulk_input,
                label="💡 Load 10 example tweets",
                examples_per_page=1,
            )

            with gr.Row():
                with gr.Column(scale=1):
                    pie_chart = gr.Image(
                        label="🥧 Sentiment Pie Chart",
                        type="pil",
                        height=380,
                    )
                with gr.Column(scale=1):
                    summary_html = gr.HTML(
                        value="<div class='empty-msg'>📊 Results will appear here after analysis.</div>",
                        label="📋 Summary & Per-Tweet Results",
                    )

            # Wiring
            bulk_btn.click(
                fn=analyze_bulk,
                inputs=bulk_input,
                outputs=[pie_chart, summary_html],
            )
            bulk_clear_btn.click(
                fn=lambda: (
                    "",
                    None,
                    "<div class='empty-msg'>📊 Results will appear here after analysis.</div>",
                ),
                inputs=None,
                outputs=[bulk_input, pie_chart, summary_html],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="footer">
        Model: distilbert-base-uncased-finetuned-sst-2-english &nbsp;·&nbsp;
        Built with 🤗 Transformers &amp; Gradio &amp; Matplotlib
    </div>
    """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
    )
