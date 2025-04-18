import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from preprocess import clean_text
from wordcloud import WordCloud

df = pd.read_csv("reviews.csv")

def initial_inspection():
	# Initial inspection
	print("Dataset Shape:", df.shape)
	print("\nFirst 5 Rows:\n", df.head())
	print("\nLast 5 Rows:\n", df.tail())
	print("\nData Types:\n", df.dtypes)
	print("\nColumn Names:", df.columns)
	print("\nBasic Info:")
	df.info()

	print("\nMissing Values per Column:\n", df.isnull().sum())
	print("\nNumber of Duplicate Rows:", df.duplicated().sum())

def analyze_each_column():
	# Single column analyzation
	# Label a.k.a star rating
	print("Star rating distribution:", df['label'].value_counts().sort_index())
	sns.countplot(x='label', data=df, palette='viridis')
	plt.title('Distribution of Star Ratings')
	plt.xlabel('Star Rating')
	plt.ylabel('Number of Reviews')
	plt.show()

	print("\nStar Rating Statistics:\n", df['label'].describe())

	print("\nSentiment Distribution:\n", df['sentiment'].value_counts())
	print("\nUnique Sentiment Values:", df['sentiment'].unique())

	sns.countplot(x='sentiment', data=df, palette='viridis', order=['Positive', 'Neutral', 'Negative']) # Control order
	plt.title('Distribution of Sentiments')
	plt.xlabel('Sentiment Label')
	plt.ylabel('Number of Reviews')
	plt.show()

	# Text
	df['review_length'] = df['text'].astype(str).apply(len) # Ensure text is string
	print("\nReview Length Statistics:\n", df['review_length'].describe())

	# Plot the distribution of review lengths
	sns.histplot(df['review_length'], bins=50, kde=True)
	plt.title('Distribution of Review Lengths (Characters)')
	plt.xlabel('Number of Characters')
	plt.ylabel('Frequency')
	plt.show()

	df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
	print("\nWord Count Statistics:\n", df['word_count'].describe())

	# Plot the distribution of word counts
	sns.histplot(df['word_count'], bins=50, kde=True)
	plt.title('Distribution of Review Word Counts')
	plt.xlabel('Number of Words')
	plt.ylabel('Frequency')
	plt.show()

def analyze_relationship_between_columns():
	df['review_length'] = df['text'].astype(str).apply(len)
	df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

	# Analyze relationshop between columns
	sns.boxplot(x='label', y='review_length', data=df, palette='viridis')
	plt.title('Review Length vs. Star Rating')
	plt.xlabel('Star Rating')
	plt.ylabel('Review Length (Characters)')
	plt.ylim(0, df['review_length'].quantile(0.99))
	plt.show()

	fig, ax = plt.subplots(1, 2, figsize=(16, 6))
	sns.boxplot(x='sentiment', y='review_length', data=df, palette='Set2', order=['Positive', 'Neutral', 'Negative'], ax=ax[0])
	ax[0].set_title('Review Length vs. Sentiment Label')
	ax[0].set_ylim(0, df['review_length'].quantile(0.99))
	sns.boxplot(x='sentiment', y='word_count', data=df, palette='Set2', order=['Positive', 'Neutral', 'Negative'], ax=ax[1])
	ax[1].set_title('Word Count vs. Sentiment Label')
	ax[1].set_ylim(0, df['word_count'].quantile(0.99))
	plt.tight_layout()
	plt.show()

	ct = pd.crosstab(df['label'], df['sentiment'], margins=True)
	print("\nRating vs. Sentiment Cross-Tabulation:\n", ct)

	ct_norm = pd.crosstab(df['label'], df['sentiment'], normalize='index')
	plt.figure(figsize=(10, 7))
	sns.heatmap(ct.drop('All', axis=1).drop('All', axis=0), annot=True, fmt='d', cmap='Blues')
	plt.title('Heatmap of Rating vs. Sentiment Counts')
	plt.show()

	plt.figure(figsize=(10, 7))
	sns.heatmap(ct_norm, annot=True, fmt='.2f', cmap='viridis')
	plt.title('Heatmap of Rating vs. Sentiment (Row Normalized - % per Rating)')
	plt.show()

	sns.boxplot(x='label', y='word_count', data=df, palette='viridis')
	plt.title('Word Count vs. Star Rating')
	plt.xlabel('Star Rating')
	plt.ylabel('Word Count')
	plt.ylim(0, df['word_count'].quantile(0.99))
	plt.show()

	print("\nAverage Length per Rating:\n", df.groupby('label')['review_length'].mean())
	print("\nAverage Word Count per Rating:\n", df.groupby('label')['word_count'].mean())

def text_content_exploration():
	df['text'] = df['text'].apply(clean_text)
	# Text content exploration
	all_reviews_text = ' '.join(df['text'].astype(str).str.lower())
	words = all_reviews_text.split()
	word_counts = Counter(words)
	print("\nTop 20 Most Common Words (Overall):\n", word_counts.most_common(20))

	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews_text)
	plt.figure(figsize=(12, 8))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')
	plt.title('Most Common Words in All Reviews')
	plt.show()

	for sentiment_val in df['sentiment'].unique():
		plot_wordcloud_for_sentiment(sentiment_val, df)

def plot_wordcloud_for_sentiment(sentiment_label, df):
	text = ' '.join(df[df['sentiment'] == sentiment_label]['text'].astype(str).str.lower())
	if not text: return

	wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
	plt.figure(figsize=(12, 8))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')
	plt.title(f'Most Common Words for {sentiment_label.capitalize()} Sentiment Reviews')
	plt.show()


if __name__ == "__main__":
	initial_inspection();
	analyze_each_column();
	analyze_relationship_between_columns();
	text_content_exploration();