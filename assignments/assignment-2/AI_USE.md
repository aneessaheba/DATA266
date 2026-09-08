# AI Use Disclosure for HW2

## 1. How I used AI

* Used Claude to help understand the concepts behind this assignment:
  * Word2Vec embeddings and how finetuning changes word meaning
  * Transfer learning for pretrained embeddings
  * How Retrieval Augmented Generation pipelines work: document loading, chunking,
    embeddings, vector stores, retrievers, and prompt templates
  * Training time optimization techniques: activation checkpointing, gradient
    accumulation, mixed precision training, and weight initialization
* Used Claude to help write and run the code for all three parts: finetuning Word2Vec on
  IMDB reviews, building the RAG pipeline with LangChain, and running the controlled
  optimization experiments
* Used Claude to build the Jupyter notebooks and this report

## 2. Something the AI got wrong

The RAG pipeline's local LLM used sampling by default, so the same question and the same
retrieved context could produce a different answer on different runs. The first run of the
Jurassic Park question answered "Velociraptor," but rerunning the exact same code could give
a different wrong answer, which meant the written analysis would not match a rerun of the
notebook. This was fixed by turning off sampling (greedy decoding) so every rerun gives the
same, reproducible answer.

## 3. How I checked everything

* I reran both notebooks end to end and confirmed the printed outputs matched every number
  written in the report
* I manually opened the Wikipedia text files and located the exact sentence containing each
  answer, to check the retrieval success table myself instead of trusting the model's output
* I compared the finetuned neighbor tables and cosine similarity numbers against the
  notebook output before writing the most shifted and least shifted conclusions
* I checked that all five optimization experiments actually finished without errors before
  recording their time, memory, and loss numbers
