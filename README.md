# 🌿 Ayurvedic Knowledge Extraction and Drug-Herb Interaction Prediction

An AI-powered system for extracting Ayurvedic knowledge from classical texts and predicting potential drug-herb interactions using Natural Language Processing (NLP), Knowledge Graphs, and Machine Learning.

## 📌 Project Overview

This project focuses on extracting medicinal knowledge from Ayurvedic literature (Charaka Samhita) and building an intelligent recommendation system that predicts drug-herb interaction risks and suggests alternative herbs when necessary.

The system combines:

- BioBERT for biomedical Named Entity Recognition (NER)
- SpaCy for text preprocessing
- Knowledge Graph construction using NetworkX
- XGBoost for interaction risk prediction
- Streamlit for the user interface

---

## 🚀 Features

- Extracts herbs, diseases, actions, properties, and side effects from Ayurvedic texts
- Builds a structured Ayurvedic Knowledge Graph
- Predicts risk levels for drug-herb combinations
- Recommends safer alternative herbs
- Interactive web application using Streamlit
- Explainable AI outputs with evidence from the knowledge base

---

## 🏗️ System Architecture

```text
Charaka Samhita PDF
        ↓
Text Extraction & Cleaning
        ↓
BioBERT + SpaCy NER
        ↓
Knowledge Graph Construction
        ↓
Drug-Herb Dataset Generation
        ↓
XGBoost Risk Prediction Model
        ↓
Alternative Herb Recommendation
        ↓
Streamlit User Interface
```

---

## 🛠️ Tech Stack

### Machine Learning & NLP

- BioBERT
- SpaCy
- XGBoost
- Scikit-learn
- Pandas
- NumPy

### Knowledge Graph

- NetworkX

### Frontend

- Streamlit

### Development Tools

- Python
- Jupyter Notebook
- Google Colab
- VS Code
- Git & GitHub

---

## 📂 Project Structure

```text
ayurvedic_knowledge_extraction/

│
├── data/
│   ├── ayurvedic_kg.json
│   ├── drug_herb_dataset.csv
│   └── herb_compounds.csv
│
├── models/
│   ├── drug_herb_xgboost_model.pkl
│   ├── feature_encoders.pkl
│   └── risk_encoder.pkl
│
├── notebooks/
│   ├── NER_Training.ipynb
│   ├── Knowledge_Graph.ipynb
│   └── Model_Training.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Preetika0/ayurvedic_knowledge_extraction.git

cd ayurvedic_knowledge_extraction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📊 Machine Learning Model

### Model Used

- XGBoost Classifier

### Input Features

- Drug Name
- Herb Name
- Compound Information
- Therapeutic Properties
- Side Effects
- Dosha Effects

### Output

- Risk Level
  - Low Risk
  - Moderate Risk
  - High Risk

- Interaction Effects

- Alternative Herb Recommendations

---

## 🌿 Example

### Input

```text
Drug: Aspirin
Herb: Pippali
```

### Output

```text
Risk Level:
High Risk

Interaction:
May increase bleeding tendencies.

Recommended Alternative:
Amalaki
```

---

## 📈 Future Enhancements

- Integration with Neo4j Graph Database
- RAG-based Ayurvedic Question Answering System
- Explainable AI dashboards
- Multi-language support for Sanskrit texts
- Larger Ayurvedic knowledge base

---

## 👨‍💻 Authors

- Y V Bhavana
- K P Preetika Setty
- Amrutha sai

---

## 📚 References

- Charaka Samhita
- BioBERT: Biomedical Language Representation Model
- XGBoost Documentation
- SpaCy NLP Library
- Scikit-learn Documentation

---

## 📄

This project is developed for educational and research purposes.

