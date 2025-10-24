import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib

lem = WordNetLemmatizer()

testing_data=pd.read_csv("test.csv")
training_data=pd.read_csv("train.csv")

stop_words = set(stopwords.words('english'))

def clean_data(data):
    data['Description'] = data['Description'].str.lower()
    data['Description'] = data['Description'].replace(r'[^\w\s]', '', regex=True)
    data['Description'] = data['Description'].replace(r'(<br\s*/?>|br\s*/?|/br)', '', regex=True)
    data['Description'] = data['Description'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    data['Description'] = data['Description'].apply(lambda x: ' '.join(lem.lemmatize(word)for word in x.split()))
    return data['Description']

clean_testing_data = clean_data(testing_data)
clean_training_data = clean_data(training_data)


x_train = clean_training_data
y_train = training_data["Class Index"]
x_test = clean_testing_data
y_test = testing_data["Class Index"]

vectorizer = TfidfVectorizer(max_features=5000)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)


model = LogisticRegression()
model.fit(x_train_vec,y_train)

y_pred = model.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "news_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model training and evaluation complete!")