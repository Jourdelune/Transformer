
from torchtext.datasets import Multi30k

from tokenizer import Tokenizer

dataset = Multi30k(split='train')
tok = Tokenizer('de_core_news_sm', 'en_core_web_sm', dataset)

for src, tgt in dataset:
    src = tok.string_to_vocab(src, True)
    print(src)

    src = tok.vocab_to_string(src, True)
    print(src)
    break
