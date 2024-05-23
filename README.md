# GPT-2 Vision

Learn how to build a vision-language model 

GPT-2 Vision is a language model that has been augmented with CLIP to understand images. Given an image, it is able to generate text describing that image. It was written and trained from scratch in under 1000 lines of pure PyTorch

## Background

### Language models

Given a sequence of words (a prompt), language models predict the next likely word

For example:

A cat sat on a red ___.

A language model will likely predict "mat" instead of "lava" because the training data likely contains many examples of cats sitting on mats but few of cats sitting on lava

### Embeddings

Language models like GPT-2 convert the prompt into word embeddings, which are points in a high-dimensional space where similar words are near one another. Regions in this space represent semantic meaning.

To predict the next word, the model could naively try to find an embedding closest to all the prompt word embeddings. However, this approach could fail for words with multiple meanings like "bat." Would the embedding space near "bat" represent the animal or the sports equipment?

Instead, language models use the full context of the prompt to refine the embeddings, transforming them to capture the complete meaning of the text. The next word is then predicted from a region in the space close to this context-aware representation

Consider these sentences:

1. The bat flew out of the ___.
2. The player swung the bat at the ___.

In the first sentence, a language model might focus on "flew out," adjusting the embeddings to create a representation of "bat" as a flying animal. This refined embedding might be close to "cave" in the semantic space, so the model may predict "cave."

In the second sentence, a model might focus on "player swung," adjusting the embeddings to represent "bat" as sports equipment, leading the model to predict "baseball."

**Language models use the full context of a piece of text to transform word embeddings, creating representations that capture the complete meaning of a sentence, resulting in accurate next-word predictions.**

*A GPT-2 language model converting a sequence of text into word embeddings, using these embeddings together to predict the next word*

### Vision language models

## Samples

## Install

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Train

Download the GPT-2 and CLIP pre-trained model weights

The training code updates a multi-layer perceptron to align visual embeddings from CLIP with word embeddings from GPT-2

Train using an NVIDIA GPU with at least 6GB of VRAM

```shell
$ python train.py
```

## Run

Generate captions on your own images
```shell
$ python generate.py
```
