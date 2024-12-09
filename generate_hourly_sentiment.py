import pandas as pd
import re
import sys
from datetime import datetime
from joblib import Parallel, delayed
from snownlp import SnowNLP

def main():
    # Ensure correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python generate_hourly_sentiment.py <inputfile> <outputfile>")
        sys.exit(1)
    
    # Read input and output file paths
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load the data
    try:
        data = pd.read_json(input_file, lines=True)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Precompile regex patterns
    pattern_newlines_tabs = re.compile(r'\s+')
    pattern_hashtags_mentions = re.compile(r'#\S+|@\S+')
    pattern_non_chinese = re.compile(r'[^\w\s\u4e00-\u9fff]')

    # Optimized cleaning function
    def clean_text_optimized(text):
        text = pattern_newlines_tabs.sub(' ', text)
        text = pattern_hashtags_mentions.sub('', text)
        text = pattern_non_chinese.sub('', text)
        return text.strip()

    # Function to calculate sentiment
    def calculate_sentiment_optimized(text):
        try:
            return SnowNLP(text).sentiments
        except:
            return None

    # Parallel processing for cleaning
    data['cleaned_content'] = Parallel(n_jobs=-1)(
        delayed(clean_text_optimized)(text) for text in data['content']
    )

    # Parallel processing for sentiment analysis
    data['sentiment_score'] = Parallel(n_jobs=-1)(
        delayed(calculate_sentiment_optimized)(text) for text in data['cleaned_content']
    )

    # Filter rows with valid sentiment scores
    data = data.dropna(subset=['sentiment_score'])

    # Group by date-hour and calculate average sentiment index
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data = data.dropna(subset=['created_at'])
    data['date_hour'] = data['created_at'].dt.strftime('%Y-%m-%d %H')

    hourly_sentiment = data.groupby('date_hour')['sentiment_score'].mean().reset_index()
    hourly_sentiment.rename(columns={'sentiment_score': 'sentiment_index'}, inplace=True)

    # Save the results to a CSV file
    try:
        hourly_sentiment.to_csv(output_file, index=False)
        print(f"Hourly sentiment data has been successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
