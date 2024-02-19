import pandas as pd


df1 = pd.read_csv('data1.csv', sep=',')
print (df1.text)

df2 = pd.read_csv('data2.csv', sep=',')
print (df2.text)

df1_selected = df1[['text', 'DIALECT', 'IRONY']]
df2_selected = df2[['text', 'DIALECT', 'IRONY']]

combined_df = pd.concat([df1_selected, df2_selected], ignore_index=True)

total_instances = combined_df.shape[0]

combined_df['DIALECT'] = combined_df['DIALECT'].str.split('/')
combined_df = combined_df.explode('DIALECT')
combined_df['IRONY'] = combined_df['IRONY'].replace({True: 1, False: 0})
dialect_mapping = {
    'MSA': 1,
    'MAGHREBI': 2,
    'LEVANTINE': 3,
    'GULF': 4,
    'EGYPTIAN': 5,
    'SUDANESE': 6
}

combined_df['DIALECT'] = combined_df['DIALECT'].replace(dialect_mapping)
combined_df = combined_df.dropna(subset=['DIALECT'])
combined_df['DIALECT'] = combined_df['DIALECT'].astype(int)

unique_dialects = combined_df['DIALECT'].unique()

print(combined_df)
print(unique_dialects)

combined_df.to_csv('dialect-identification.csv', sep='\t')