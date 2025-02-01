import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import json

# Load the processed dataset
data = pd.read_csv("datasets/processed_dataset.csv")
print("Loaded processed data:")
print(data.head())

# Fill missing values and ensure strings
data['Message'] = data['Message'].fillna('').astype(str)

# Verify there are no NaN values and all are strings
assert data['Message'].isnull().sum() == 0, "Missing values still present in 'Message'"
assert data['Message'].apply(lambda x: isinstance(x, str)).all(), "Non-string values found in 'Message'"

# Split into features and labels
X = data['Message']
y = data['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()
joblib.dump(tfidf,"models/tfidfVectorizer.joblib")
print("model saved as")


# Models and Evaluation
model_metrics={}
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear'),
    #"SVM": SVC(kernel='linear',probability=True),   #for svm probability
    "Naive Bayes": MultinomialNB()
}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Save the trained model
    model_file = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_file)
    print(f"models/{model_name} model saved as '{model_file}'")
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate
    print(f"\n{model_name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    #saving metrics
    accuracy=accuracy_score(y_test, y_pred)
    precision = classification_report(y_test, y_pred, output_dict=True)['macro avg']['precision']
    recall = classification_report(y_test, y_pred, output_dict=True)['macro avg']['recall']
    f1_score = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']
    
    #json file
    model_metrics[model_name] = {
        "accuracy": f"{accuracy:.4f}",
        "precision": f"{precision:.4f}",
        "recall": f"{recall:.4f}",
        "f1": f"{f1_score:.4f}"
    }
    #saving model metrics
    with open('models/model_metrics.json', 'w') as json_file:
        json.dump(model_metrics, json_file, indent=4)
    
    #confusion matrix saving 
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted spam/ham')
    plt.ylabel('Actual spam/ham')
    path=f'static/{model_name}confusion_matrix.png'
    print(f"Confusion Matrix saved as '{path}'")
    plt.savefig(path)
    plt.close()


