from data_processing import data_processing
from train import regression_model
from errors import test_model, generate_classification_report, generate_error_file


train_messages, train_class_labels, test_messages, test_class_labels, valid_messages, valid_class_labels = data_processing()
model = regression_model(train_messages, train_class_labels)
predictions, label_encoder = test_model(model, valid_messages, valid_class_labels)
generate_classification_report(valid_class_labels, predictions, label_encoder)
generate_error_file(valid_messages, valid_class_labels, predictions, label_encoder)