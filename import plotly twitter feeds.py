import plotly.graph_objs as go
import json
import nltk
import random
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist

# Download the NLTK datasets (if not already downloaded)
nltk.download("twitter_samples")
nltk.download("punkt")
nltk.download("stopwords")

# Load positive and negative tweets from the NLTK Twitter dataset
positive_tweets = twitter_samples.strings("positive_tweets.json")
negative_tweets = twitter_samples.strings("negative_tweets.json")

# Tokenize tweets into words
positive_tweet_tokens = [word_tokenize(tweet) for tweet in positive_tweets]
negative_tweet_tokens = [word_tokenize(tweet) for tweet in negative_tweets]

# Define stopwords and punctuation
stop_words = set(stopwords.words("english"))
punctuation = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!']

# Function to preprocess tweets (remove stopwords and punctuation)
def preprocess_tweet(tweet_tokens):
    cleaned_tokens = []
    for token in tweet_tokens:
        if token.lower() not in stop_words and token not in punctuation:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

# Preprocess positive and negative tweets
positive_cleaned_tokens_list = [preprocess_tweet(tokens) for tokens in positive_tweet_tokens]
negative_cleaned_tokens_list = [preprocess_tweet(tokens) for tokens in negative_tweet_tokens]

# Combine positive and negative tokens for further analysis
all_cleaned_tokens = positive_cleaned_tokens_list + negative_cleaned_tokens_list

# Create a list of all words
all_words = [word for tokens in all_cleaned_tokens for word in tokens]

# Calculate word frequency distribution
freq_dist = FreqDist(all_words)

# Function to extract features from a tweet (using the most common words as features)
def get_features(tweet_tokens, common_words):
    features = {}
    for word in common_words:
        features[word] = word in tweet_tokens
    return features

# Define the most common words as features (adjust the number as needed)
common_words = [word[0] for word in freq_dist.most_common(20)]

# Create a feature set for each tweet
positive_features = [(get_features(tweet_tokens, common_words), "Positive") for tweet_tokens in positive_cleaned_tokens_list]
negative_features = [(get_features(tweet_tokens, common_words), "Negative") for tweet_tokens in negative_cleaned_tokens_list]

# Combine positive and negative feature sets
features = positive_features + negative_features

# Shuffle the feature sets
random.shuffle(features)

# Train a Naive Bayes classifier
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(features)

# Test the classifier
accuracy = nltk.classify.util.accuracy(classifier, features)
print("Accuracy:", accuracy)

# Test the classifier on custom tweets
custom_tweet = input("Enter a custom tweet: ")
custom_tokens = preprocess_tweet(word_tokenize(custom_tweet))
custom_features = get_features(custom_tokens, common_words)
sentiment = classifier.classify(custom_features)
print("Sentiment:", sentiment)
