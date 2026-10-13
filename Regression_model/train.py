from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


def regression_model(train_messages, train_class_labels):

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 1)),  # You can change this to (2,2) for bi-grams or (1,2) for uni-grams and bi-grams
        LogisticRegression(max_iter=1000,C=7)
    )
    model.fit(train_messages, train_class_labels)
    return model
