# Infosys_Internship
 Disaster Tweet Analyzer

## Overview
Disaster Tweet Analyzer is a machine learning-powered web application designed to classify tweets as disaster-related or non-disaster-related. The application leverages Natural Language Processing (NLP) techniques to extract relevant information such as locations, disaster categories, and sentiment from tweets, aiding in disaster management and response efforts.

## Features
- **Tweet Classification**: Determines if a tweet is disaster-related or not.
- **Location Extraction**: Identifies locations mentioned in tweets using NLP and regex-based methods.
- **Disaster Category Detection**: Recognizes disaster categories (e.g., flood, earthquake, wildfire) based on predefined keywords.
- **Sentiment Analysis**: Analyzes the sentiment of tweets (Positive, Negative, or Neutral).

## Project Structure
- **`app1.py`**: Main Flask application file containing routes for prediction, location extraction, disaster categorization, and sentiment analysis.
- **`lr_model.pkl`**: Pre-trained Logistic Regression model used for disaster tweet classification.
- **`scaler.pkl`**: Scaler object used for normalizing vectorized tweet data.
- **`vectorizer.pkl`**: Trained vectorizer for transforming textual data into numerical features.
- **`train.txt`**: A text file containing information about disaster types, tweet analysis steps, and relevant disaster concepts.

## Installation and Setup
1. Clone or download the repository.

2. Install the required Python libraries:
   - Flask
   - spaCy
   - NLTK
   - scikit-learn
   - pickle

## Technologies Used

- **Programming Language**: Python  
- **Web Framework**: Flask  
- **NLP Tools**: spaCy, NLTK  
- **Machine Learning**: scikit-learn  
- **Models Used**: Logistic Regression  
- **Text Processing**: Tokenization, Vectorization, Scaling  

## Future Enhancements

- Integration with Twitter API for real-time tweet analysis.  
- Support for multilingual tweet classification.  
- Addition of fake news detection for disaster-related tweets.  
- Improved scalability and real-time prediction capabilities.  

