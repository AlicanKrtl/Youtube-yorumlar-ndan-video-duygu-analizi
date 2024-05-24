import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Display the first few rows of the DataFrame to understand its structure
print(df.head())

# Get basic statistics of the 'emotion' column
emotion_stats = df['emotion'].value_counts()

# Plot a bar chart to visualize the distribution of emotions
plt.figure(figsize=(8, 6))
emotion_stats.plot(kind='bar', color='skyblue')
plt.title('Distribution of Emotions')
plt.xlabel('Emotion')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Get basic statistics of the 'comments' column
comments_stats = df['comments'].describe()

# Print the statistics of the 'comments' column
print(comments_stats)
