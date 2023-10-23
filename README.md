# Simple Copilot for Specialized Question-Answering

## About

This project is an implementation of [OpenAI's Search and Ask Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb). It serves as a specialized Copilot capable of answering specific questions based on provided data sets. The system is versatile, allowing for the incorporation of data ranging from Mobile App Reviews to Corporate Handbooks.

## Features
- Answer questions in natural language
- Uses OpenAI's embeddings for precise search
- Uses OpenAI's chat/completions for answering questions
- Extensible to various data sources and types
- Easily configurable for different use-cases

## How it works
OpenAI embeddings convert documents and queries into vector representations for comparison. They map text and code to vectors in a high-dimensional space, with closer embeddings indicating similar data. Practical applications include search, clustering and recommendations.

We can expand to chat-based applications using the search-and-ask method:

* **Search**: A knowledge base is formed with document embeddings for each section. When a user queries, the question is converted into a query embedding to find relevant sections in the knowledge base.
* **Ask**: This relevant information both results and the user query is then used to create a prompt to generate user responses.

## Dependencies

Run the following command to install the necessary packages:
```bash
pip3 install -U argparse openai pandas yaml scipy tiktoken
```

## Quick Start

1. **Generate Embeddings:** First, you'll need to create embeddings for your text data.
    ```bash
    python generate_embeddings.py --config config-custom.yml
    ```
2. **Query the System:** Once the embeddings are ready, start querying.
    ```bash
    python search_and_ask_embeddings.py --config config-custom.yml --query "Your question goes here"
    ```

## Data Preparation

Place your `.csv` or `.tsv` data files under the `data/` folder. It's advisable to use a single row for each complete piece of information, such as a sentence or definition. Afterward, generate the embeddings to enable natural language querying.

## Configuration

Create different configuration YAML files tailored to various data sets and use-cases. This modular approach allows you to reuse the same source code while easily switching between different configurations.

## Helpful Resources

- [OpenAI Cookbook Example](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)
- [Introduction to Text Embeddings with OpenAI](https://www.datacamp.com/tutorial/introduction-to-text-embeddings-with-the-open-ai-api)
- [OpenAI Embeddings Use-Cases](https://platform.openai.com/docs/guides/embeddings/use-cases)
- [Semantic Text Search Using Embeddings](https://cookbook.openai.com/examples/semantic_text_search_using_embeddings)

## License

This project is open source, under the [Unlicense](LICENSE).

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) to get started.
