import os
import sys
import re
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nlkit.utils import get_linear_schedule_with_warmup_ep, weight_init

sys.path.append("../..")
from modern_poetry.trainer import Trainer
from modern_poetry.model import Generator
from modern_poetry.data import Dataset
from modern_poetry.vocab import Vocab
from modern_poetry.utils import load_modern_poems, translate_logits, load_ancient_poems
from modern_poetry.sampler import Sampler
logging.basicConfig(level=logging.INFO)

from flask import Flask, render_template, request

app = Flask(__name__)


def load_sampler(max_len, model_path):
    max_seq_len = max_len

    num_embeddings = 4865
    embedding_dim = 256
    hidden_dim = 256
    rnn_layers = 3
    
    sampling_topk = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    wv_path = os.path.join(base_dir, "..", "./resource/w2v/word.wv")
    model_path = os.path.join(base_dir, "..", model_path)
    
    sampler = Sampler(
        max_decode_len=max_seq_len, sampling_topk=sampling_topk,
        device=device, wv_path=wv_path, model=None, model_path=model_path,
        num_embeddings=num_embeddings, embedding_dim=embedding_dim,
        hidden_dim=hidden_dim, rnn_layers=rnn_layers,
        
    )

    print("sampler loaded")
    
    return sampler


modern_sampler = load_sampler(500, "./output/model.mod")
ancient_sampler = load_sampler(300, "./output/model.anc")


def inference(sampler):
    hint = request.form["hint"]
    prompts = [["<sos>"] + list(hint)]
    print(prompts)
    infer_result = sampler.sample(prompts)
    infer_result = infer_result[0][5:]

    infer_result = re.split("[，。？！；：、]", infer_result)
    result = infer_result
    print(result)
    
    return render_template('query.html', GenerateText=result)


@app.route("/")
def index():
    return render_template('query.html')


@app.route("/modern/", methods=['POST'])
def modern():
    return inference(modern_sampler)


@app.route("/ancient/", methods=['POST'])
def ancient():
    return inference(ancient_sampler)
