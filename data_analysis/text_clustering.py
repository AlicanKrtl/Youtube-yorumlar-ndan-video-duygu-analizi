import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the CSV data into a pandas DataFrame
df = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

# Extract comments from the DataFrame
comments = df['comments']

# Initialize the TfidfVectorizer to convert comments into TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english', lowercase=True)
tfidf_matrix = tfidf_vectorizer.fit_transform(comments)

# Determine the optimal number of clusters using the silhouette score
max_clusters = 20
best_score = -1
best_n_clusters = 6

for n_clusters in range(2, max_clusters+1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

# Perform KMeans clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

# Add cluster labels to the DataFrame
df['cluster'] = cluster_labels

# Print the top terms per cluster
print("Top terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorizer.get_feature_names_out()
for i in range(best_n_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print()
