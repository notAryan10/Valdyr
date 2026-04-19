## Live Application

🔗 https://housepredict112.streamlit.app/

---
# 🏠 Intelligent Property Price Advisory System

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Nandann018-ux/Valdyr/main?labpath=notebooks%2Fdata_exploration.ipynb)

## Overview
This project is an AI-driven real estate analytics system that predicts property prices using historical housing data and provides structured buying or investment recommendations.

It combines machine learning for price prediction with an agent-based advisory module to assist users in making informed real estate decisions.

---

## Objectives
- Predict property prices or price ranges  
- Identify key price-driving factors  
- Provide investment/buying insights  
- Build an interactive user interface  
- Extend into an agentic AI advisory assistant  

---

## Dataset
- Kaggle Housing Prices Dataset  
- Dataful Real Estate Dataset  

Features include location, size, rooms, age, and amenities.

---

## Tech Stack
- **Languages:** Python
- **Libraries:** pandas, NumPy, scikit-learn
- **Frontend:** Streamlit
- **AI/ML:** LangGraph / LangChain
- **Vector DB:** FAISS / ChromaDB (optional)

---

## Features
- Property data upload  
- Price prediction  
- Model performance metrics  
- Advisory report generation  
- Market insight summaries  

---

## Project Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Jupyter notebooks for data exploration and model training.
- `src/`: Source code for the application.
- `requirements.txt`: Python dependencies.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook / Lab

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Nandann018-ux/Valdyr.git
    cd Valdyr
    ```

2.  **Create a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**1. Create Environment Variables**
Create a `.env` file in the root directory to store your API keys:
```bash
GROQ_API_KEY=your_groq_key_here
```

**2. Run the Web Application**
Launch the Streamlit dashboard to predict prices and generate AI advisory reports:
```bash
streamlit run app/streamlit_app.py
```

**3. Run Jupyter Lab (Optional Data Exploration)**
```bash
jupyter lab
```
Navigate to `notebooks/data_exploration.ipynb` to explore the dataset and models.
