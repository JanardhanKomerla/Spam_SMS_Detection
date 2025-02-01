import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

data=pd.read_csv('datasets/combined_file.csv')

#checking null values
print(data.isnull().sum())
print(data.head())

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [PorterStemmer().stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Apply preprocessing
data['Message'] = data['Message'].apply(preprocess_text)
print(data.head())
ph= "datasets/processed_dataset.csv"
data.to_csv(ph, index=False)
print(f"Processed data saved to {ph}")