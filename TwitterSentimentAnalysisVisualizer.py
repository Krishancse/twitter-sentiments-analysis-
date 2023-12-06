a Python script that generates various types of plots and visualizations using the Plotly library based on the provided functions, we can modify your existing code and integrate the Plotly visualization functions. Additionally, I'll include the WordCloud visualization. Make sure to install the required libraries first:

```bash
pip install plotly wordcloud
```
import plotly.graph_objs as go
import json
from wordcloud import WordCloud
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
common_words = freq_dist.most_common(20)
common_words = [word[0] for word in common_words]

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

# Function to generate timeline graph
def timeline_graph(tweets):
    dates = tweets.groupby(by='date').count()['text']

    data = [go.Bar(
        x=dates.index,
        y=dates
    )]

    layout = go.Layout(title={'text': 'User Timeline',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=go.PlotlyJSONEncoder)
    return graphJSON

# Function to generate source graph
def source_graph(tweets):
    source = tweets.groupby('source').count()['text'].sort_values(ascending=False)

    data = [go.Bar(
        x=source.index,
        y=source
    )]

    layout = go.Layout(title={'text': 'Tweet Sources',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=go.PlotlyJSONEncoder)
    return graphJSON

# Function to generate active week graph
def active_week_graph(tweets):
    active = tweets.groupby('day').count()['text']
    active = active.reindex(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

    data = [go.Bar(
        x=active.index,
        y=active
    )]

    layout = go.Layout(title={'text': 'Day of Week',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=go.PlotlyJSONEncoder)
    return graphJSON

# Function to generate active hour heatmap
def active_hr_heatmap(tweets):
    time = [x for x in range(24)]
    day = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    z = [[tweets[(tweets['day'] == x) & (tweets['time'].apply(lambda x: x.hour) == y)].count()['text'] for y in time]
         for x in day]

    hovertext = list()
    for yi, yy in enumerate(day):
        hovertext.append(list())
        for xi, xx in enumerate(time):
            hovertext[-1].append('Hour: {}<br />Day: {}<br />Tweets: {}'.format(xx, yy, z[yi][xi]))

    data = [go.Heatmap(z=z, x=time, y=day, colorscale='Reds', hoverinfo='text', text=hovertext)]
    layout = go.Layout(title={'text': 'Daily Rhythm',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                       xaxis=dict(title='Hour',
                                  tick0=0,
                                  dtick=1,
                                  ticklen=24,
                                  tickwidth=1),
                       yaxis=dict(title='Day'))

    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=go.PlotlyJSONEncoder)
    return graphJSON

# Function to generate tweet type graph
def tweet_type_graph(tweets):
    typ = tweets.groupby('type').count()['text'].sort_values()

    data = [go.Bar(
        x=typ,
        y=typ.index,
        orientation='v'  # Assuming you want the orientation to be 'vertical'. Adjust as needed.
    )]

    layout = go.Layout(title={'text': 'Tweet Types',
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'})
    fig = go.Figure(data=data, layout=layout)

    graphJSON = json.dumps(fig, cls=go.PlotlyJSONEncoder)
    return graphJSON
