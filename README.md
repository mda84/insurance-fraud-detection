# Insurance Claim Fraud Detection using AI & LLMs

This repository demonstrates an AI-driven project for insurance fraud detection by combining traditional machine learning with state-of-the-art LLM-based text analysis. The solution leverages structured claim data along with unstructured claim descriptions to predict fraudulent claims in real time and provide textual explanations using an LLM.

## Repository Structure
```
insurance-fraud-detection/
├── README.md
├── requirements.txt
├── notebooks
│   ├── Fraud_Detection_EDA.ipynb
│   ├── Model_Training_and_Evaluation.ipynb
│   └── LLM_Explanation_Demo.ipynb
├── src
│   ├── data_download.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── llm_integration.py
│   ├── model_training.py
│   └── utils.py
└── api
    └── app.py
```

## Setup Instructions

1. Clone the Repository:
```
git clone https://github.com/yourusername/insurance-fraud-detection.git
cd insurance-fraud-detection
```

2. Create and Activate a Virtual Environment:
```
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

3. Install Dependencies:
```
pip install -r requirements.txt
```

4. Download and Prepare the Dataset:

Update the URL in src/data_download.py with the actual dataset URL.
Run the script to download the dataset:
```
python src/data_download.py
```
This will save the dataset to data/claims.csv.

5. Explore and Run Notebooks:

Open notebooks/Fraud_Detection_EDA.ipynb for exploratory data analysis.
Open notebooks/Model_Training_and_Evaluation.ipynb to train and evaluate models.
Open notebooks/LLM_Explanation_Demo.ipynb to see how the LLM generates claim explanations.

6. Run the API:
```
uvicorn api.app:app --reload
```
The API exposes endpoints /predict for fraud prediction and /explain for generating claim explanations.