import pandas as pd
from simpletransformers.seq2seq import (
    Seq2SeqModel,
    Seq2SeqArgs,
)
from sklearn.model_selection import train_test_split



df = pd.read_csv('../data/edited_irony_data.csv', sep=',')
df = df.dropna(subset=['phase'])

num_rows_column = df['phase'].count()
print(num_rows_column)
# 500 count

#remove the words with @
mask = df['text'].str.contains('@')
df = df[~mask]
df['text'] = df['text'].str.replace(r'http\S+', '', regex=True)


trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
train_df = trainset[['text', 'phase']]

eval_df = testset[['text', 'phase']]

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="facebook/bart-large",
    args=model_args,
    use_cuda=True,
)


def count_matches(labels, preds):
    print(labels)
    print(preds)
    return sum(
        [
            1 if label == pred else 0
            for label, pred in zip(labels, preds)
        ]
    )


# Train the model
model.train_model(
    train_df, eval_data=eval_df, matches=count_matches
)

# # Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction
# print(
#     model.predict(
#         [
#             "Tyson is a Cyclops, a son of Poseidon, and Percy Jackson’s half brother. He is the current general of the Cyclopes army."
#         ]
#     )
# )
