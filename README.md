# Real/Fake Job Prediction

This project aims to predict whether a job posting is fraudulent or not using machine learning techniques. It utilizes various features from job postings to train an XGBoost classifier and evaluate its performance.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project follows these main steps:
1. Load and explore the dataset
2. Perform exploratory data analysis
3. Preprocess the text data
4. Extract features using TF-IDF
5. Train an XGBoost classifier
6. Perform hyperparameter tuning
7. Evaluate the model's performance

## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- nltk
- spacy
- xgboost
- wordcloud
- imbalanced-learn

You can install the required packages using:

```
pip install pandas numpy seaborn matplotlib scikit-learn nltk spacy xgboost wordcloud imbalanced-learn
```

Additionally, you'll need to download the English language model for spaCy:

```
python -m spacy download en_core_web_sm
```

## Data

The project uses a dataset named `job_train.csv`. Ensure this file is in the same directory as the script or update the file path accordingly.

## Exploratory Data Analysis

The EDA includes:
- Distribution of job characteristics (telecommuting, company logo, questions)
- Word clouds for job titles and requirements
- Distribution of job locations
- Proportion of fraudulent vs. non-fraudulent job postings

## Data Preprocessing

The preprocessing steps include:
- Handling missing values
- Text cleaning (lowercasing, removing HTML tags, URLs, and non-alphanumeric characters)
- Removing stop words
- Lemmatization
- Removing punctuation

## Model Training

The project uses an XGBoost classifier with hyperparameter tuning via GridSearchCV. The features are extracted using TF-IDF vectorization.

## Evaluation

The model is evaluated using:
- Classification report (precision, recall, F1-score)
- Accuracy score
- Confusion matrix
- ROC curve and AUC score

## Usage

1. Ensure all dependencies are installed
2. Place the `job_train.csv` file in the project directory
3. Run the script:

```
python real_fake_job_prediction.py
```

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
