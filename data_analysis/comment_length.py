import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Calculate the length of each comment
df['comment_length'] = df['comments'].apply(lambda x: len(" ".join(eval(x)).split()))

# Calculate average text length and standard deviation
avg_length = df['comment_length'].mean()
std_dev = df['comment_length'].std()

# Print the statistics
print("Average Text Length:", avg_length)
print("Standard Deviation of Text Length:", std_dev)
