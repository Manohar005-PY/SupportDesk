from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from data_processing import data_processing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# config = {
#     "uni_gram": (1,1),
#     "bi_gram": (2,2),
#     "uni_gram_and_bi_gram": (1,2),
# }
# unigram preforemed well so we will use that model and we will use the logistic regression model with different values of C to see which one performs best.
# refer comparison__report.txt for the results of the different models and their performance metrics.


train_messages, train_labels, test_messages, test_labels, valid_messages, valid_labels = data_processing()
# for c in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
# resluts for above values of C are as follows:
# Model: C=0.1
# Accuracy: 0.7208
# F1 Score: 0.6970
# Model: C=0.5
# Accuracy: 0.8503
# F1 Score: 0.8471
# Model: C=1.0
# Accuracy: 0.8773
# F1 Score: 0.8772
# Model: C=2.0
# Accuracy: 0.8929
# F1 Score: 0.8930
# Model: C=5.0
# Accuracy: 0.8997
# F1 Score: 0.8999
# Model: C=10.0
# Accuracy: 0.9000
# F1 Score: 0.9001

# we try to narrow the c value to find the best performing model and we will use the logistic regression model with different values of C to see which one performs best.
for c in [7, 10, 15, 20, 30]:

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1,1)), # You can change this to (2,2) for bi-grams or (1,2) for uni-grams and bi-grams
        LogisticRegression(max_iter=2000, C=c)
    )
    model.fit(train_messages, train_labels)
    predictions = model.predict(valid_messages)

    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    valid_labels_encoded = label_encoder.transform(valid_labels)
    predictions_encoded = label_encoder.transform(predictions)

    accuracy = accuracy_score(valid_labels_encoded, predictions_encoded)
    f1 = f1_score(valid_labels_encoded, predictions_encoded, average='weighted')

    print(f"Model: C={c}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
