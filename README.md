# Transformer implementation
My implementation of the transformer architecture from the paper [Attention is all you need](https://arxiv.org/abs/1706.03762).

# Why an another implementation?

I made this code to learn the basis of pytorch and practice my skill in deep learning. There is a lot of chance that the implementation is wrong so I do not recommend using it, it's just a student project.

# What I use to build the project

I used the "Attention is all you need" paper but also a lot of external resources like:
- https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch
- https://github.com/hyunwoongko/transformer
- https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- https://gmongaras.medium.com/how-do-self-attention-masks-work-72ed9382510f

You can take a look on the resource.

# Training

You just have to follow these two steps

1. Install modules:
`pip install -r requirements.txt`

2. Run the training script: `python3 train.py`

# TODO

- [ ] Implement learning rate scheduler
- [ ] Add script to run prediction
- [ ] Use sentencepiece for tokenization
