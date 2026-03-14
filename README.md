Twitter Sentiment Analysis Dashboard
This project is a high-performance Natural Language Processing (NLP) application designed to analyze the sentiment of social media text. Built using Hugging Face Transformers and Gradio, it provides both real-time single-sentence analysis and bulk data visualization through interactive pie charts.

🚀 Key Features
Real-Time Analysis: Instant POSITIVE/NEGATIVE classification with confidence scoring.

Bulk Processing: Analyze multiple tweets or sentences simultaneously by pasting text line-by-line.

Data Visualization: Automatically generates a Matplotlib Pie Chart to visualize the overall mood distribution of bulk data.

Modern UI: A clean, dual-tabbed web interface powered by Gradio.

🧠 The Tech Stack
Model: distilbert-base-uncased-finetuned-sst-2-english (A distilled, fast version of BERT).

Framework: Hugging Face Transformers & PyTorch.

Interface: Gradio.

Visualization: Matplotlib & Pandas.

🛠️ Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/TalhaNoor12/SoftGrowTech_Twitter_Sentiment_Analysis.git
cd SoftGrowTech_Twitter_Sentiment_Analysis
Create a Virtual Environment:

Bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the App:

Bash
python3 app.py
📸 Demo Screenshots
Include your screenshots here in your GitHub repo!

Single Prediction: [Link to screenshot]

Bulk Analysis Pie Chart: [Link to screenshot]

📜 SoftGrow Tech Internship
This project was developed as part of the Artificial Intelligence Internship at SoftGrow Tech. It demonstrates proficiency in NLP pipelines, UI development, and data visualization.