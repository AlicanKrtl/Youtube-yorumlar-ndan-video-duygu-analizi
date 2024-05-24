import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Combine all comments into a single string
all_comments = ' '.join(df['comments'])

# Split the string into words
words = all_comments.split()

# Count the frequency of each word
word_freq = Counter(words)

# Convert the word frequency counter to a DataFrame for easier manipulation
word_freq_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])

# Sort the DataFrame by frequency in descending order
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

# Display the top 10 most frequent words
print(word_freq_df.head(10))

# Plot the word frequency distribution
plt.figure(figsize=(10, 6))
plt.bar(word_freq_df['Word'][:20], word_freq_df['Frequency'][:20], color='skyblue')
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
