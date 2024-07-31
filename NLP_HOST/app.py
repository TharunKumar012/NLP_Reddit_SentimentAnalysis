from flask import Flask, request, render_template, redirect, url_for, send_file
import praw
import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from io import BytesIO
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns

nltk.download('stopwords')
app = Flask(__name__)

reddit = praw.Reddit(
    client_id="RUdBseyl4eBxlRaQWPgKYA",
    client_secret="jz3MH5GDM7LnYIGnZlb7vICnglEuhw",
    user_agent="my-app by u/JPIERCE115"
)

def extract_comment_data(comment):
    return {
        'id': comment.id,
        'author': comment.author.name if comment.author else "Deleted",
        'body': comment.body,
        'score': comment.score,
        'created_utc': comment.created_utc
    }

def most_freq_words(comments_df):
    text = ' '.join(comments_df['body'])
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
    
    fig_size = (12,6)  # Width and height adjusted proportionally
    plt.figure(figsize=fig_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return img

def vader_sentiment_analysis(comments_df):
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment(text):
        scores = analyzer.polarity_scores(text)
        return scores['compound']
    
    comments_df['sentiment'] = comments_df['body'].apply(get_sentiment)
    
    # Histogram
    fig_size = (10,6)  # Width and height adjusted proportionally
    plt.figure(figsize=fig_size)
    sns.histplot(comments_df['sentiment'], bins=30, kde=True)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    sentiment_hist_path = os.path.join(extraction_folder, 'sentiment_histogram.png')
    plt.savefig(sentiment_hist_path)
    plt.close()
    
    # Bar plot
    def classify_sentiment(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    comments_df['sentiment_category'] = comments_df['sentiment'].apply(classify_sentiment)
    
    plt.figure(figsize=fig_size)
    palette = {
        'positive': 'green',
        'neutral': 'blue',
        'negative': 'red'
    }
    ax = sns.countplot(data=comments_df, x='sentiment_category', order=['positive', 'neutral', 'negative'], palette=palette)
    plt.title('Sentiment Category Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')

    # Add y-axis values on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 3), 
                    textcoords='offset points')

    sentiment_bar_path = os.path.join(extraction_folder, 'sentiment_bar.png')
    plt.savefig(sentiment_bar_path)
    plt.close()

    return sentiment_hist_path, sentiment_bar_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_comments', methods=['POST'])
def extract_comments():
    post_url = request.form['post_url']
    post_id = post_url.split('/')[-3]

    # Get the post
    post = reddit.submission(id=post_id)

    # Fetch all comments
    post.comments.replace_more(limit=None)
    comments = post.comments.list()

    # Extract data from comments
    comments_data = [extract_comment_data(comment) for comment in comments]

    # Create a DataFrame
    comments_df = pd.DataFrame(comments_data)

    # Convert the 'created_utc' to a readable datetime format
    comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], unit='s')

    # Define the path to the extraction folder inside NLP_HOST
    global extraction_folder
    extraction_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extraction')

    # Ensure the extraction directory exists
    if not os.path.exists(extraction_folder):
        os.makedirs(extraction_folder)

    # Save the DataFrame to a CSV file in the extraction folder
    comments_df.to_csv(os.path.join(extraction_folder, 'comments.csv'), index=False)

    return render_template('extracted.html')

@app.route('/post_analysis', methods=['POST'])
def post_analysis():
    # Load the CSV file
    comments_df = pd.read_csv(os.path.join(extraction_folder, 'comments.csv'))
    
    # Convert the 'created_utc' column to datetime format
    comments_df['created_utc'] = pd.to_datetime(comments_df['created_utc'], errors='coerce', unit='s')
    
    # Handle any errors in conversion
    if comments_df['created_utc'].isnull().any():
        print("Warning: Some 'created_utc' values could not be converted to datetime.")
    
    # Generate the word cloud and save it
    img = most_freq_words(comments_df)
    with open(os.path.join(extraction_folder, 'wordcloud.png'), 'wb') as f:
        f.write(img.getvalue())

    # Perform VADER sentiment analysis and save the visualizations
    vader_sentiment_analysis(comments_df)

    return redirect(url_for('display_visualizations'))

@app.route('/display_visualizations')
def display_visualizations():
    return render_template('visualizations.html')

@app.route('/image/<image_name>')
def image(image_name):
    return send_file(os.path.join(extraction_folder, image_name), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
