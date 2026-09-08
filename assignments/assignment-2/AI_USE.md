# AI Use Disclosure for HW2

## 1. How I used AI

* Used Claude to help me learn the concepts behind this assignment:
  * Word2Vec embeddings and how finetuning changes word meaning
  * Transfer learning for pretrained embeddings
  * How Retrieval Augmented Generation pipelines work: document loading, chunking,
    embeddings, vector stores, retrievers, and prompt templates
  * Training time optimization techniques: activation checkpointing, gradient
    accumulation, mixed precision training, and weight initialization

## 2. Something the AI got wrong

When explaining how to measure the memory savings from activation checkpointing, the AI
first suggested measuring memory once per training step, after the optimizer update. This
turned out to be wrong: memory measured at that point looked the same whether checkpointing
was on or off, because by then the intermediate activations had already been freed either
way. The correct explanation is that the memory difference only shows up if it is measured
right after the forward pass, before backward propagation frees the activations. Once I
understood this, the memory numbers matched what the concept predicted.

## 3. How I checked everything

* I checked the memory numbers from my own notebook run against what I learned about how
  checkpointing and gradient accumulation are supposed to behave, to confirm the numbers
  matched the concept
* I looked up the ground truth passage in the Wikipedia text myself for each RAG question,
  to check the retrieval success table
* I compared the neighbor tables and cosine similarity numbers against what I learned about
  how finetuning is supposed to shift embeddings, before writing the most shifted and least
  shifted conclusions
* I confirmed all five optimization experiments finished without errors before recording
  their time, memory, and loss numbers
