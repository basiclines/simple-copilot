import os                                           # Read file extensions
import argparse                                     # Read command line arguments
import yaml                                         # Read .yaml files
import openai                                       # Calling the OpenAI API
import pandas as pd                                 # Storing text and embeddings data in .tsv/.csv
from scipy import spatial                           # Calculating vector similarities for search
from ast import literal_eval                        # Converting embeddings saved as strings back to arrays
from openai.embeddings_utils import get_embedding
import tiktoken                                     # Counting tokens
import sys                                          # handle command line parameters
from colorama import Fore, Back, Style              # Colored text output

# Define arguments for command line
parser = argparse.ArgumentParser(description='Process some named arguments.')
parser.add_argument('--config', type=str, help='Your config file')
parser.add_argument('--query', type=str, help='Your question to ask in natural language')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
  config = yaml.safe_load(f)

# Assign config values
openai.api_key = config['openai']['key']
datafile_path = config['data']['dist']
openai_embed_model = config['openai']['embedding_model']
openai_chat_model = config['openai']['chat_model']
system_prompt = config['prompt']['system']
prompt_introduction = config['prompt']['user']
result_prompt_intro = config['prompt']['present_result']
text_column = config['data']['text_column']
embedding_column = config['data']['embedding_column']
prompt_budget = config['prompt']['budget']

# Assign default separator
file_extension = os.path.splitext(datafile_path)[1]
if file_extension == '.tsv':
  separator = '\t'
elif file_extension == '.csv':
  separator = config['data'].get('separator', ',')
else:
  raise ValueError("Unsupported file format, please use .tsv or .csv")

# read file with pre-computed text and embeddings
df = pd.read_csv(datafile_path, sep=separator)

# convert embeddings from CSV str type back to list type
df[embedding_column] = df[embedding_column].apply(literal_eval)

# search function
def get_results_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    print(Fore.YELLOW + "Searching for related entries...")
    query_embedding_response = openai.Embedding.create(
        model=openai_embed_model,
        input=query,
    )
    query_embedding = query_embedding_response["data"][0]["embedding"]
    strings_and_relatednesses = [
        (row[text_column], relatedness_fn(query_embedding, row[embedding_column]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

def count_tokens(text: str) -> int:
  """Return the number of tokens in a string."""
  encoding = tiktoken.encoding_for_model(openai_chat_model)
  return len(encoding.encode(text))

def build_prompt(
    query: str,
    df: pd.DataFrame,
    token_budget: int
) -> str:
  """Return a message for GPT, with relevant source texts pulled from a dataframe."""
  strings, relatednesses = get_results_ranked_by_relatedness(query, df)
  question = f"\n\nQuestion: {query}"
  message = prompt_introduction
  for string in strings:
    next_article = f'\n\n{result_prompt_intro}\n"""\n{string}\n"""'
    if (
      count_tokens(message + next_article + question)
      > token_budget
    ):
      break
    else:
      message += next_article
  return message + question

def search(query):
  strings, relatednesses = get_results_ranked_by_relatedness(query, df, top_n=5)
  for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)

def ask(
    query: str,
    df: pd.DataFrame = df,
    token_budget: int = prompt_budget - 500,
    print_message: bool = False,
) -> str:
  message = build_prompt(query, df, token_budget=token_budget)
  messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": message},
  ]
  print(Fore.YELLOW + "Asking to GPT...")
  response = openai.ChatCompletion.create(
    model=openai_chat_model,
    messages=messages,
    temperature=0
  )
  response_message = response["choices"][0]["message"]["content"]
  print('-' * os.get_terminal_size().columns)
  print('\n')
  print(Fore.BLUE + '💬 Your question:')
  print(Fore.BLUE + args.query)
  print('\n')
  print(Fore.GREEN + '🤖 GPT response:')
  print(Fore.GREEN + response_message)
  print('\n')
  print('-' * os.get_terminal_size().columns)

ask(args.query)
