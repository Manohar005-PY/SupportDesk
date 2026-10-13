# Banking Intent Classification

This project trains a text classifier to predict the intent category of a
banking-related customer message. Although the code folder is named
`Regression_model`, the task is **multi-class classification**, not regression.

## Model

The training pipeline in `Regression_model/train.py` uses:

- A unigram TF-IDF vectorizer (`ngram_range=(1, 1)`) to represent message text.
- Scikit-learn `LogisticRegression` with `C=7` and `max_iter=1000` to predict
  one of the dataset's 77 intent categories.

`Regression_model/data_processing.py` reads the Banking dataset's `text` and
`category` columns. It splits the upstream training data into an 80% training
set and a 20% validation set using `random_state=42`. The upstream test CSV is
loaded by the data-processing function, but the current `main.py` evaluates
against the validation split instead.

The saved classification report records **0.89 accuracy** and **0.89 weighted
F1** on 2,001 validation examples. These are validation results, not results on
the upstream held-out test set. Running the program writes a classification
report and misclassified validation examples under `Regression_model/`.

## Dataset

The dataset is not included in this repository. Download it separately into
the project root under `task-specific-datasets/`:

```powershell
git clone https://github.com/PolyAI-LDN/task-specific-datasets.git
```

The model uses the `banking_data/train.csv` and `banking_data/test.csv` files
from that repository.

### Dataset credit

The Banking dataset is provided by **PolyAI-LDN**. Please credit the authors
of the associated paper:

> Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan
> Vulić. “Efficient Intent Detection with Dual Sentence Encoders.” Proceedings
> of the 2nd Workshop on NLP for ConvAI, ACL 2020.

- Dataset repository: [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)
- Paper: [arXiv:2003.04807](https://arxiv.org/abs/2003.04807)
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as
  specified by the upstream repository.

## Run

From the project root, install the dependencies and run the training script:

```powershell
python -m pip install -r requirements.txt
Set-Location Regression_model
python main.py
```
