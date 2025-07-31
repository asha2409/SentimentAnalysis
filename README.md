# SentimentAnalysis
**Sentiment Analysis of Tweets**
This project focuses on analyzing the sentiment of tweets, classifying them as either non-racist/sexist (label 0) or racist/sexist (label 1). The process involves data cleaning, exploratory data analysis, text vectorization using TF-IDF, and training a Logistic Regression model for classification.

**Dataset**
The dataset used in this project consists of tweets, provided in two CSV files:

train_E6oV3lV.csv: Contains the training data with tweet IDs, labels (0 or 1), and the tweet text.
test_tweets_anuFYb8.csv: Contains the test data with tweet IDs and the tweet text.
**Project Steps**
**The project follows these main steps:**
*Data Loading and Merging:
*Load the train and test datasets and combine them for consistent preprocessing.
*Data Cleaning:
Remove Twitter handles (@user).
Remove punctuation, numbers, and special characters (keeping hashtags).
Convert tweets to lowercase.
Remove short words (length 3 or less).
Exploratory Data Analysis (EDA):
Visualize the distribution of tweet lengths.
Generate Word Clouds to understand the most frequent words in all tweets, non-racist/sexist tweets, and racist/sexist tweets.
Analyze the frequency of hashtags in both categories of tweets.
Visualize co-occurrence of words in racist/sexist tweets using a heatmap.
Text Vectorization: Convert the cleaned tweet text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
*Model Training:
Split the data into training and validation sets.
Train a Logistic Regression model on the TF-IDF features.
Model Evaluation: Evaluate the performance of the Logistic Regression model using metrics like accuracy, confusion matrix, and classification report.
*Prediction and Submission:
Clean the test data using the same cleaning steps.
Vectorize the cleaned test tweets using the same TF-IDF vectorizer fitted on the training data.
Predict the sentiment labels for the test data.
Create a submission file in the specified format (id, label).
Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical operations.
re: For regular expressions used in text cleaning.
nltk: For natural language processing tasks, including tokenization and frequency distribution.
string: For string operations.
seaborn and matplotlib.pyplot: For data visualization.
wordcloud: To generate word clouds.
sklearn: For machine learning models (TF-IDF, Logistic Regression, train-test split, metrics).
gensim: For Word2Vec and Doc2Vec (although TF-IDF was primarily used for the final model).
tqdm: For progress bars.
How to Run the Notebook
Upload the train_E6oV3lV.csv and test_tweets_anuFYb8.csv files to your Colab environment.
Run the cells sequentially. The notebook is structured to perform data loading, cleaning, EDA, vectorization, model training, and prediction in order.
Submission File
The output of the prediction on the test data is saved as submission_logreg.csv, which is formatted for submission to platforms like Analytics Vidhya.

Author
Shaik Asha Muskan

