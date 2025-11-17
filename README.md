# AdIntel
AdIntel is an AI platform that forecasts ad performance, provides statistically backed A/B test analysis, and generates unique user behavior insights to drive smarter marketing decisions. This is a project I built to move beyond simple marketing analytics and create an intelligent tool for ad performance. The goal of AdIntel is to replace guesswork with data-driven certainty.

I built a full-stack web app that combines two major features:

A Live CTR Predictor that uses a high-performance AI (0.73 AUC) to forecast an ad's success.

A Statistical A/B Test Analyzer to definitively prove which ad version is the winner.

This project took me from raw data exploration and advanced feature engineering all the way to a final, deployed multi-page application.

(Action: Take a screenshot of your dashboard, upload it to GitHub, and replace this link!)

✨ What It Does: Core Features

I wanted to build an app that felt complete and professional, so I focused on a few key features:

📱 Interactive Multi-Page Application: I designed the app with a clean Streamlit sidebar to make it easy to switch between the prediction and analysis tools.

📊 Built-in Demo Mode: The A/B Test Analyzer loads a sample dataset on startup, so anyone can see it work instantly without needing to find a CSV file.

⚡ Optimized Performance: The large AI model (.pkl file) and all the encoding maps are loaded once and cached, so all future predictions are instantaneous.

🔮 Live CTR Predictor: This is the AI-powered part. It's a simple form where you can input an ad's features and my model will give you a real-time click probability score.

🔬 Statistical A/B Test Analyzer: This tool lets a user upload their own campaign data. It runs a Chi-Squared test to provide a clear, statistical verdict, preventing costly decisions based on random chance.

🚀 The AI Model (0.73 AUC)

Getting a high score was a major goal. The final model (a LightGBM classifier) achieved a 0.73 AUC / 0.37 LogLoss on a 3-million-row sample of the Avazu dataset.

The key were the custom features I engineered:

Advanced Target Encoding: Instead of using raw categories, I devised features like device_id and app_category. This was done after splitting the data to prevent data leakage (the "illusion trap" which I had encountered)

🧠 Behavioral Features: I also created features to model how a real user behaves like

user_ad_count: Tracks ad fatigue.

user_historical_ctr: Calculates a user's personal click rate (this was tricky to build without leakage!).

time_since_last_ad: Models user session timing.

🛠️ Tech Stack

Data Science & ML: Python, Pandas, NumPy, Scikit-learn

Machine Learning Model: LightGBM

Model Serving: Joblib (for saving/loading the model & encoders)

Web Application: Streamlit

Statistical Analysis: SciPy

Environment: Google Colab (for the heavy-duty model training), VS Code

🏁 How to Run This Project

Clone the Repository:

git clone [https://github.com/umairaalvi4843-hub/AdIntel.git](https://github.com/umairaalvi4843-hub/AdIntel.git)
cd AdIntel


Create and Activate a Virtual Environment:

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate


Install Dependencies:

# Install all required libraries
pip install -r requirements.txt


Run the Streamlit App:
(The required .pkl model assets are included in this repo for demo purposes)

# This is the most reliable way to run the app
python -m streamlit run app.py


Your browser will automatically open to the AdIntel dashboard.
