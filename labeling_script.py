import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load your dataset
# Replace 'dataset.csv' with your file name
df = pd.read_csv('dataset.csv')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to classify sentiment
def get_label(text):
    if pd.isna(text):
        return 'neutral'
    
    score = analyzer.polarity_scores(str(text))['compound']
    
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Create the new label column
df['label'] = df['text'].apply(get_label)

# Save the labeled dataset
df.to_csv('labeled_dataset.csv', index=False)

print("Dataset labeled successfully!")
print(df.head())
