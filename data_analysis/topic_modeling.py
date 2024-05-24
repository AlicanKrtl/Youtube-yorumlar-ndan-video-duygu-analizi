from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Combine all comments into a single string
text = ' '.join(df['comments'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Comments')
plt.axis('off')
plt.show()
