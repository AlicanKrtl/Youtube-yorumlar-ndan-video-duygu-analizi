import os
import pandas as pd
from googleapiclient.discovery import build
from dataset.data_keys import return_search_keys
from tqdm import tqdm

# Set up the API key and build the YouTube service
API_KEY = 'AIzaSyABpKJNRdw7duYKePXzfXwGLFTgqP1XMfk'  # Replace with your actual API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Function to search for videos
def search_videos(query, max_results=1):
    request = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video'
    )
    response = request.execute()
    return response['items']

# Function to get video details including top comments
def get_video_details(video_id, comment_num=5):
    video_details = {}

    # Get video statistics
    video_request = youtube.videos().list(
        part='snippet,statistics',
        id=video_id
    )
    video_response = video_request.execute()
    if video_response['items']:
        video = video_response['items'][0]
        video_details['title'] = video['snippet']['title']
        video_details['url'] = f"https://www.youtube.com/watch?v={video_id}"
        video_details['views'] = video['statistics']['viewCount']
    
    # Get top comments
    comments_request = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=comment_num,
        order='relevance'
    )
    comments_response = comments_request.execute()
    top_comments = []
    for item in comments_response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        top_comments.append({
            'text': comment['textOriginal'],
            'likes': comment['likeCount']
        })
    video_details['top_comments'] = top_comments
    
    return video_details

def get_comments(search_key:str, max_results:int, comment_num:int, emotion:str):
    video_items = search_videos(search_key, max_results)

    video_data = []
    for item in video_items:
        try:    
            video_id = item['id']['videoId']
            video_details = get_video_details(video_id, comment_num)
            video_details['emotion'] = emotion  # Add the emotion to the video details
            video_data.append(video_details)
        except:
            continue
    if len(video_data) != 0:
        df = pd.DataFrame(video_data)
        df["top_comments"] = df['top_comments'].apply(lambda x: [list(i.values())[0] for i in x])
        url_comment = df[["url", "top_comments", "emotion"]]
    else:
        url_comment = pd.DataFrame(columns=["url", "top_comments", "emotion"])
    return url_comment


if __name__ == "__main__":
    search_keys = return_search_keys()

    # Create an empty dataframe to store all results
    all_results = pd.DataFrame()

    for emotion, query_list in search_keys.items():
        for query in tqdm(query_list):
            df = get_comments(query, 20, 20, emotion)
            all_results = pd.concat([all_results, df], ignore_index=True)
    
    # Save the combined results to a CSV file
    all_results.to_csv(f"youtube_comments_disgust1.csv", index=False)