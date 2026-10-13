import pandas as pd
from sklearn.model_selection import train_test_split
def data_processing():
    raw_data = pd.read_csv(r'D:\SupportDesk\model_train\task-specific-datasets\banking_data\train.csv', encoding='latin-1')
    raw_test_data = pd.read_csv(r'D:\SupportDesk\model_train\task-specific-datasets\banking_data\test.csv', encoding='latin-1')
    train_messages, valid_messages, train_class_labels, valid_class_labels = train_test_split(
        raw_data['text'].values.astype('U'),
        raw_data['category'].values.astype('U'),
        test_size=0.2,
        random_state=42
    )
    test_messages = raw_test_data['text'].values.astype('U')
    test_class_labels = raw_test_data['category'].values.astype('U')

    return train_messages, train_class_labels, test_messages, test_class_labels, valid_messages, valid_class_labels
