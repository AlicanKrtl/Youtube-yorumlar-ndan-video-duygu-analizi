from textblob import TextBlob
import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Function to perform sentiment analysis on a text
def analyze_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get the sentiment polarity (-1 for negative, 0 for neutral, 1 for positive)
    sentiment_polarity = blob.sentiment.polarity
    # Classify sentiment based on polarity
    if sentiment_polarity < 0:
        return 'Negative'
    elif sentiment_polarity == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Apply sentiment analysis to each comment in the DataFrame
df['sentiment'] = df['comments'].apply(analyze_sentiment)

# Display the sentiment analysis results
print(df['sentiment'].value_counts())
