import os
import argparse
import openai
import pandas as pd
import yaml

# Define arguments for command line
parser = argparse.ArgumentParser(description='Process some named arguments.')
parser.add_argument('--config', type=str, help='Your config file')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
  config = yaml.safe_load(f)

# Assign config values
openai.api_key = config['openai']['key']
read_file = config['data']['src']
write_file = config['data']['dist']
embedding_model = config['openai']['embedding_model']
text_column = config['data']['text_column']
embedding_column = config['data']['embedding_column']

# Assign default separator
file_extension = os.path.splitext(read_file)[1]
if file_extension == '.tsv':
  separator = '\t'
elif file_extension == '.csv':
  separator = config['data'].get('separator', ',')
else:
  raise ValueError("Unsupported file format, please use .tsv or .csv")

# Get Embedding for a given line of text
def get_embedding(text_to_embed):

  print("Start embeeding for {}".format(text_to_embed))
  response = openai.Embedding.create(
      model= embedding_model,
      input=[text_to_embed]
  )
  # Extract the AI output embedding as a list of floats
  embedding = response["data"][0]["embedding"]
  print("Finished embeeding: {}".format(embedding))

  return embedding

# Load CSV and define text column
review_df = pd.read_csv(read_file, sep=separator)
review_df.head()
review_df = review_df[[text_column]]

# Read CSV line by line and obtain embeddings
review_df[embedding_column] = review_df[text_column].astype(str).apply(get_embedding)

# Save embeddings
review_df.to_csv(write_file, sep=separator, index=False)

