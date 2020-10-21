import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gluonnlp as nlp
# MXNet-based
import mxnet as mx
# PyTorch-based
import transformers
import torch
from transformers import BertModel, BertConfig
from gluonnlp.model import get_model as _get_model
from mxnet.gluon import Block

from pytorch_block_sparse import BlockSparseModelPatcher


from .bert import BERTRegression, AlbertForMaskedLMOptimized, BertForMaskedLMOptimized, DistilBertForMaskedLMOptimized
from .gpt2 import gpt2_117m, gpt2_345m


# get_model() is from:
# https://github.com/dmlc/gluon-nlp/blob/master/scripts/text_generation/model/__init__.py
def get_model(name: str, **kwargs) -> Tuple[Block, nlp.Vocab]:
    """Returns a pre-defined model by name.

    In addition to the models in GluonNLP model API, this API supports getting GPT-2 models.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        For language model, options are 'wikitext-2'.
        For ELMo, Options are 'gbw' and '5bw'.
        'gbw' represents 1 Billion Word Language Model Benchmark
        http://www.statmt.org/lm-benchmark/;
        '5bw' represents a dataset of 5.5B tokens consisting of
        Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B).
        If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    models: Dict[str, Block] = {
        'gpt2_117m': gpt2_117m,
        'gpt2_345m': gpt2_345m
    }
    name = name.lower()
    if name not in models:
        return _get_model(name, **kwargs)
    return models[name](**kwargs)


# Shortcodes for MXNet models
# These should not conflict w/ HuggingFace Transformer's shortcodes

SUPPORTED_MLMS = [
    'bert-base-en-uncased',
    'bert-base-en-cased',
    'roberta-base-en-cased',
    'bert-large-en-uncased',
    'bert-large-en-cased',
    'roberta-large-en-cased',
    'bert-base-en-uncased-owt',
    'bert-base-multi-uncased',
    'bert-base-multi-cased'
]

SUPPORTED_LMS = [
    'gpt2-117m-en-cased',
    'gpt2-345m-en-cased'
]

SUPPORTED = SUPPORTED_MLMS + SUPPORTED_LMS


def get_pretrained(ctxs: List[mx.Context], name: str = 'bert-base-uncased',
                   cased: bool = False, finetune: bool = False, regression: bool = False, freeze: int = 0,
                   root: Optional[Path] = None) -> Tuple[Block, nlp.Vocab, nlp.data.BERTTokenizer]:

    model = sparsify_and_load(name)
    model.save_pretrained("./tmp")
    model = BertForMaskedLMOptimized.from_pretrained("./tmp")
    tokenizer = transformers.BertTokenizer.from_pretrained(name)
    return model, None, tokenizer


def sparsify_and_load(name: str = 'bert-base-uncased', lh=(4, 256), density=0.5):

    config = BertConfig(hidden_size=lh[1], num_attention_heads=int(lh[1] / 64),
                        num_hidden_layers=lh[0], intermediate_size=4 * lh[1])

    model = BertModel(config)
    mp = BlockSparseModelPatcher()
    mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.intermediate\.dense", {"density": density})
    mp.add_pattern("bert\.encoder\.layer\.[0-9]+\.output\.dense", {"density": density})
    mp.patch_model(model.cuda())

    model.load_state_dict(torch.load(name))
    return model
