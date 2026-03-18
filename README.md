🧠 ChurnSense AI
“Upload. Understand. Act.”

An interactive platform for understanding and acting on customer churn data using a full suite of analytical tools, from exploratory analysis to AI-driven action recommendations.

ChurnSense AI is a full-stack web application that allows users to upload a customer dataset and receive a comprehensive analysis, including machine learning predictions, causal inference, survival analysis, and reinforcement learning recommendations to reduce churn.

✨ Features Implemented
This platform provides a complete, end-to-end workflow for churn analysis:

📤 Upload Dataset: Upload customer data in .csv format. The backend automatically detects key columns like the churn indicator and time-based features.

🧪 Exploratory Data Analysis (EDA): Statistical summaries, missing value visualization, and interactive correlation heatmaps. Includes a downloadable PDF report.

🤖 Churn Prediction (ML): Automatically trains and compares models (Logistic Regression, Random Forest, XGBoost). Displays Accuracy, Precision, Recall, and ROC-AUC.

⏳ Survival Analysis: Uses Kaplan-Meier and Cox Proportional Hazards to predict when churn is likely to happen and identify high-risk customers.

🔎 Causal Inference: Uses the DoWhy library to move beyond correlation and understand the causal effect of specific features (e.g., monthly fees) on churn.

🎮 Action Recommendation (RL): Uses a Deep Q-Network (DQN) to learn an optimal policy. Input a customer profile to receive specific recommendations (e.g., "Offer Promo," "Send Email").

🛠️ Tech Stack
Layer	Technology
Frontend	React.js, Tailwind CSS
Backend	FastAPI (Python), Uvicorn
Machine Learning	Scikit-learn, XGBoost
Deep Learning (RL)	PyTorch
Causal Inference	DoWhy
Survival Analysis	Lifelines
Data Handling	Pandas, NumPy
📂 Project Structure
Plaintext
churnsense-ai2/
├── backend/
│   ├── main.py              # FastAPI entry point
│   ├── eda_utils.py         # Data processing & EDA
│   ├── ml_utils.py          # Model training & evaluation
│   ├── survival_utils.py    # Survival analysis logic
│   ├── causal_utils.py      # Causal graphs & inference
│   ├── rl_agent.py          # DQN Agent architecture
│   ├── rl_environment.py    # Custom Gym environment
│   └── uploads/             # User-uploaded CSVs
├── frontend/
│   ├── src/
│   │   └── App.js           # Main React component
│   └── public/
├── models/                  # Saved .pkl and .pt models
└── plots/                   # Generated analysis visualizations
🚀 Getting Started
1. Clone the Repository

Bash
git clone https://github.com/sarayu2005/churnsense-ai2.git
cd churnsense-ai2
2. Backend Setup

We recommend using a virtual environment to manage dependencies.

Bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
The backend will be running at http://127.0.0.1:8000.

3. Frontend Setup

Bash
cd ../frontend
npm install
npm start
The application will open at http://localhost:3000.

📋 How to Use
Upload Data: Click "Upload CSV Dataset" and select your customer data file.

Run Analyses: Use the navigation tabs to run EDA, Survival Analysis, ML Prediction, and Causal Inference.

Train the RL Agent: Once the ML model is trained, click "Train RL Agent" to initialize the DQN training process.

Get Recommendations: Input specific customer details into the form to receive an AI-powered action recommendation.
