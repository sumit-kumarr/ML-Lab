import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load and prepare data
msg = pd.read_csv('naivetext.csv', names=['Message', 'Label'], encoding='latin1')
msg['Label'] = msg['Label'].fillna(0)
msg['labelnum'] = LabelEncoder().fit_transform(msg['Label'])
X, y = msg['Message'], msg['labelnum']

# Print dataset info
print(f"Dataset shape: {msg.shape}")
print(f"Label counts: {pd.Series(y).value_counts()}")

# Split without stratify since we have too few samples in some classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Vectorize text
count_vect = CountVectorizer(min_df=1)
X_train_dtm = count_vect.fit_transform(X_train)
X_test_dtm = count_vect.transform(X_test)

# Train Naive Bayes
clf = MultinomialNB(alpha=1.0).fit(X_train_dtm, y_train)
y_pred = clf.predict(X_test_dtm)

# Calculate metrics
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.3f}")
print(f"Confusion Matrix:\n{metrics.confusion_matrix(y_test, y_pred)}")

# Handle potential division by zero in metrics
try:
    avg = 'weighted'
    precision = metrics.precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average=avg, zero_division=0)
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
except Exception as e:
    print(f"Could not calculate some metrics: {e}")