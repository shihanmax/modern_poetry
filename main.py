import sys
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nlkit.utils import get_linear_schedule_with_warmup_ep, weight_init

sys.path.append("..")
from modern_poetry.trainer import Trainer
from modern_poetry.model import Generator
from modern_poetry.data import Dataset
from modern_poetry.vocab import Vocab
from modern_poetry.utils import load_modern_poems, translate_logits, load_ancient_poems
logging.basicConfig(level=logging.INFO)

num_embeddings = 4865
embedding_dim = 256
hidden_dim = 256

test_ratio = 0.1
valid_ratio = 0.1

batch_size = 3
lr = 1e-3
num_warmup_epochs = 3
epochs = 50

gradient_clip = 10
not_early_stopping_at_first = 10
es_with_no_improvement_after = 10
verbose = 40

max_seq_len = 100
wv_path = "./resource/w2v/word.wv"
modern_poems_path = "./resource/modern/"
ancient_poems_path = "./resource/ancient/"
model_path = "./output/model.ep"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vocab = Vocab(wv_path)

model = Generator(
    embeddings=vocab.embedding.to(torch.float),
    num_embeddings=num_embeddings,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    device=device,
)

all_poems = load_ancient_poems(ancient_poems_path)[10000:10200]
# all_poems = load_modern_poems(modern_poems_path)[:30]

all_length = len(all_poems)
test_num = round(test_ratio * all_length)
valid_num = round(valid_ratio * all_length)
test_poems = all_poems[:test_num]
valid_poems = all_poems[-valid_num:]
train_poems = all_poems[test_num: -valid_num]

train_dataset = Dataset(
    str2idx=vocab.str2idx, 
    sos_idx=vocab.sos, 
    eos_idx=vocab.eos, 
    pad_idx=vocab.pad, 
    unk_idx=vocab.unk, 
    max_seq_len=max_seq_len, 
    all_poems=train_poems,
)

valid_dataset = Dataset(
    str2idx=vocab.str2idx, 
    sos_idx=vocab.sos, 
    eos_idx=vocab.eos, 
    pad_idx=vocab.pad, 
    unk_idx=vocab.unk, 
    max_seq_len=max_seq_len, 
    all_poems=valid_poems,
)

test_dataset = Dataset(
    str2idx=vocab.str2idx, 
    sos_idx=vocab.sos, 
    eos_idx=vocab.eos, 
    pad_idx=vocab.pad, 
    unk_idx=vocab.unk, 
    max_seq_len=max_seq_len, 
    all_poems=test_poems,
)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

optimizer = AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    amsgrad=True,
)

lr_scheduler = get_linear_schedule_with_warmup_ep(
    optimizer, num_warmup_epochs, epochs, last_epoch=-1,
)

trainer = Trainer(
    model=model, 
    train_data_loader=train_data_loader, 
    valid_data_loader=valid_data_loader, 
    test_data_loader=test_data_loader, 
    lr_scheduler=lr_scheduler, 
    optimizer=optimizer, 
    weight_init=weight_init, 
    summary_path=None, 
    device=device, 
    criterion=None,
    total_epoch=epochs, 
    model_path=model_path, 
    gradient_clip=gradient_clip,
    not_early_stopping_at_first=not_early_stopping_at_first,
    es_with_no_improvement_after=es_with_no_improvement_after, 
    verbose=verbose,
    vocab=vocab,
    max_decode_len=max_seq_len,
    idx2str=vocab.idx2str, 
    str2idx=vocab.str2idx
)

trainer.start_train()
