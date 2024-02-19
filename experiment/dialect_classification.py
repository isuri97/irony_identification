from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import logging


# Load data
parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--epochs', required=False, default= 4)
arguments = parser.parse_args()

df = pd.read_csv('../data/dialect-identification.csv', sep='\t')
trainset, testset = train_test_split(df, test_size=0.2, random_state=42)

train_df = trainset[['text', 'DIALECT']]
test_sentences = testset['text'].tolist()

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
EPOCHS = int(arguments.epochs)

# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": EPOCHS,
             "train_batch_size": 8,
             "use_multiprocessing": False,
             "use_multiprocessing_for_evaluation":False,
             "n_fold":1,
             "wandb_project":"SAD",
             # "use_early_stopping ":True,
              # "evaluate_during_training":True,
              # "save_best_model":True,
              "learning_rate": 4e-5
        }

model = ClassificationModel(
    'bert',
    'bert-base-cased',
    args=train_args
)

# Train the model
# model.train_model(train_df, eval_df=val_df)
model.train_model(train_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_sentences)
print(predictions)

