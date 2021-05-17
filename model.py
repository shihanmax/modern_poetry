import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Generator(nn.Module):
    
    def __init__(
        self, num_embeddings, embedding_dim, hidden_dim, rnn_layers, device,
        pretrained_embeddings=None,
    ):
        super(Generator, self).__init__()
        self.device = device
        
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        
        if pretrained_embeddings:
            self.embedding = self.embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0,
            )
        
        self.lstm = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
        )
        
        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=num_embeddings,
            bias=False,
        )
        
        self.relu = nn.ReLU()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward_rnn(self, src, valid_length):
        _, max_src_len = src.shape[:2]
        
        embedding = self.embedding(src)  # bs, max_src_len, embedding_dim
        
        packed_repr = pack_padded_sequence(
            embedding, valid_length.cpu().to(torch.int64), batch_first=True,
            enforce_sorted=False,
        )

        # output: [bs, max_src_len, hidden * 2]
        # hn, cn: [2, bs, hidden]
        output, hidden = self.lstm(packed_repr)

        output, _ = pad_packed_sequence(
            output, batch_first=True, total_length=max_src_len,
        )
        
        output = self.fc(output)
        
        return output, hidden

    def forward_loss(self, src, tgt, valid_length):
        # bs, max_len, vocab_size
        output, _ = self.forward_rnn(src, valid_length)
        output = output.transpose(-1, -2)
        
        loss = self.cross_entropy_loss(output, tgt)
        
        return loss
    
    def forward_decoding(
        self, hint_token_ids, valid_length, max_decode_len, sampling_topk=-1,
        ignore_token_ids=(),
    ):
        """Decode by given tokens.

        Args:
            hint_token_ids (Tensor): bs, some_len
        """
        bs, hint_len = hint_token_ids.shape
        
        result = []

        _, hidden = self.forward_rnn(hint_token_ids, valid_length)
        
        inp = hint_token_ids[:, -1].unsqueeze(1)

        while len(result) < max_decode_len - hint_len:
            inp = self.embedding(inp)
            curr_out, hidden = self.lstm(inp, hidden)
            curr_out = self.fc(curr_out)
            
            # sampling
            if sampling_topk > 0:
                max_k = curr_out.shape[-1]
                top_k = min(max_k, sampling_topk)
                topk_probs, topk_indices = torch.topk(
                    curr_out[:, 0, :], k=top_k, dim=-1,
                )
                
                topk_probs /= torch.sum(topk_probs, dim=-1, keepdim=True)
                
                # add prob mask to special tokens (like <unk>, ...)
                mask = torch.as_tensor(topk_probs).int()
                for ignore_id in ignore_token_ids:
                    mask |= (topk_indices == ignore_id)

                topk_probs = topk_probs.masked_fill(mask, value=-1e9)
                
                topk_probs = torch.softmax(topk_probs, dim=-1)
                topk_probs = topk_probs.cpu().numpy()
                topk_indices = topk_indices.cpu().numpy()
                
                selected = []
                
                for i in range(bs):
                    prob = topk_probs[i]
                    indice = topk_indices[i]
                    
                    selected.append(
                        torch.from_numpy(
                            np.random.choice(indice, size=1, p=prob)
                        )
                    )
                
                inp_idx = torch.stack(selected, dim=1)  # bs, 1
                result.append(inp_idx.view(bs))
                inp = inp_idx.view(bs, 1)
            
            # greedy
            else:
                inp_idx = torch.argmax(curr_out[:, 0, :], dim=-1)
                inp_idx = inp_idx.squeeze()
                result.append(inp_idx.view(bs))

                inp = inp_idx.view(bs, 1)  # bs, 1
            
        result = torch.stack(result, dim=1)  # bs, max_decode_len - hint_len
        result = torch.cat([hint_token_ids, result], dim=1)

        return result
