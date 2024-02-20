from sklearn import metrics
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import torch
import logging


# Load data
parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-base-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--epochs', required=False, default= 3)
arguments = parser.parse_args()

df = pd.read_csv('../data/dialect-identification.csv', sep='\t')
trainset, testset = train_test_split(df, test_size=0.2, random_state=42)

train_df = trainset[['text', 'IRONY']]
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
             # "use_early_stopping ":True,
              # "evaluate_during_training":True,
              # "save_best_model":True,
              "learning_rate": 4e-5,
        }

cuda_available = torch.cuda.is_available()
model = ClassificationModel(
    MODEL_TYPE, MODEL_NAME,
    args=train_args,
    use_cuda=cuda_available,
)

# Train the model
model.train_model(train_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_sentences)
print(predictions)

results, outputs, preds_list, truths, preds, words = model.eval_model(testset)
classification_report_str = metrics.classification_report(truths,preds,digits=4)

with open('output.txt', 'w') as output_file:
    output_file.write(classification_report_str)

