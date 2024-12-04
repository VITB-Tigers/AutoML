# Automated Machine Learning (AutoML)

This is the main branch of AutoML repository.

---

## Table of Contents
- [Project Description](#project-description)
- [Directory Structure](#directory-structure)
- [Program Flow](#program-flow)
- [Steps to Run](#steps-to-run)

---

## Project Description

AutoML makes building, training, and optimizing machine learning models easy by handling tricky tasks like feature selection, model picking, and hyperparameter tuning—so you can focus on looking smart. This project is a beginner-friendly AutoML [MVP](https://en.wikipedia.org/wiki/Minimum_viable_product) to help you learn the ropes and feel like a data science pro in no time.

---

## Directory Structure
```
├── Code\
│   ├── app.py                  # Main Streamlit web application for the project.
│   ├── db_utils.py             # Helper functions for interacting with MongoDB.  
│   ├── ingest.py               # Handles data ingestion and storage in MongoDB.
│   ├── ml_utils.py             # Functions to implement and use ML models.
│   └── utils.py                # General-purpose utilities like YAML handling and file operations.
├── Models\
│   ├── ada_boost_model.pkl     # Pre-trained AdaBoost model saved as a pickle file.
│   ├── decision_tree_model.pkl # Pre-trained Decision Tree model saved as a pickle file.
│   └── linear_model.pkl        # Pre-trained Linear Regression model saved as a pickle file.
├── .gitignore                  # Files and directories to be ignored by Git.
├── config.yaml                 # Configuration file for project settings.
├── README.md                   # Documentation describing the project purpose, usage, and structure.
└── requirements.txt            # Dependencies required for the project
```

---

## Program Flow

1. **`db_utils`:** Contains utility functions to connect to MongoDB database, create tables, and insert data into them.
2. **`ingest.py`:** Ingests data from a file and store it in MongoDB.
3. **`ml_utils`:** Contains functions to train, evaluate, and save the machine learning models.
4. **`utils`:** Contains general utility functions for the project.
5. **`app.py`** Creates a web application using Streamlit to process data and train, evaluate, and save three different ML models: AdaBoost, Decision Tree, and Linear Model.

---

## Steps to Run

1. Clone the repository:
    ```
    git clone https://github.com/VITB-Tigers/AutoML
    ```

2. Create and activate a virtual environment (recommended). If using Conda:
    ```
    conda create -n env_name python==3.12.0 -y
    conda activate env_name
    ```
    > *This project uses Python 3.12.0*

3. Install the necessary packages:
    ```
    pip install -r requirements.txt
    ```

4. Run the Streamlit web application:
    ```
    streamlit run Code/app.py
    ```