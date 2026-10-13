from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def test_model(model,test_messages,test_class_labels):
    predictions = model.predict(test_messages)

    label_encoder = LabelEncoder()
    label_encoder.fit(test_class_labels)
    predictions = label_encoder.transform(predictions)
    return predictions, label_encoder

def generate_classification_report(test_class_labels, predictions,label_encoder):
    report = classification_report(
        y_true = label_encoder.transform(test_class_labels),
        y_pred = predictions,
        target_names = label_encoder.classes_
    )
    with open("classification_report_Regression.txt", "w") as f:
        f.write(report)

    print("Report Has been generated and saved as classification_report_Regression.txt")
    print(report)

def generate_error_file(test_messages,test_class_labels,predictions,label_encoder):

    errors = pd.DataFrame(
        {
            "message": list(test_messages),
            "actual": list(test_class_labels),
            "predictions": list(label_encoder.inverse_transform(predictions))
        }
    )
    errors = errors[errors["actual"] != errors["predictions"]]
    errors.to_csv("errors_Regression.csv", index=False)
    print("Error file has been generated and saved as errors_Regression.csv")