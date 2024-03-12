Dataset link: https://www.kaggle.com/competitions/quora-question-pairs

# Quora Question Similarity Project

This project aims to predict the similarity between two questions on Quora using Natural Language Processing (NLP) methods. The goal is to assist in identifying duplicate or highly similar questions, which can be valuable for content moderation and user experience improvement on the Quora platform.

## Overview

Quora is a popular platform where users ask questions, answer them, and engage in discussions on various topics. However, due to the vast amount of content generated, there is a need to identify and manage duplicate or highly similar questions. This project addresses this need by leveraging NLP techniques to build a predictive model that determines the similarity between pairs of questions.

## Dataset

The dataset used for this project consists of pairs of questions from Quora, along with labels indicating whether they are similar or not. The dataset is preprocessed and cleaned to remove noise and irrelevant information, ensuring the quality of the training data.

## Approach

1. **Data Preprocessing**: Tokenization, lowercasing, removing stop words, and other preprocessing techniques are applied to clean the text data.

2. **Feature Extraction**: Various features are extracted from the text, including word embeddings, TF-IDF vectors, and syntactic features.

3. **Model Building**: Several machine learning and deep learning models are explored and evaluated for their effectiveness in predicting question similarity. Random Forests is used for trained and compared.

4. **Evaluation**: The performance of the models is evaluated using metrics such as accuracy, precision, recall, and F1-score. Cross-validation and hyperparameter tuning techniques are employed to optimize model performance.

## Usage

1. **Dependencies**: Ensure that all required libraries and dependencies are installed.

2. **Data**: Prepare your dataset in the same format as the provided Quora question pairs dataset or modify the data loading functions accordingly.

3. **Training**: Train the model using the provided scripts or notebooks. Experiment with different models and hyperparameters to achieve the best performance.

4. **Evaluation**: Evaluate the trained model using appropriate evaluation metrics to assess its effectiveness in predicting question similarity.


## Future Work

- Incorporate more advanced NLP techniques such as BERT or GPT-based models for improved performance.
- Explore ensemble methods to combine predictions from multiple models for enhanced accuracy.
- Enhance the model's scalability and efficiency to handle larger volumes of data and real-time processing.

## Contributors

- [Hrishabh598] - [Developer]

Feel free to contribute to this project by submitting pull requests or opening issues for any suggestions or improvements. Happy coding!
