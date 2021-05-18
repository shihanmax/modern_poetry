import torch

from .utils import translate_logits
from .model import Generator
from .vocab import Vocab


def do_sample(
    model, vocab, str2idx, prompts, max_decode_len, sampling_topk, device,
):
    model.eval()

    prompts_ids = [
        [vocab.str2idx.get(i, str2idx["<unk>"]) for i in prompt]
        for prompt in prompts
    ]
    valid_length = torch.tensor([len(i) for i in prompts_ids])

    prompts_token_ids = torch.tensor(prompts_ids).to(device)
    
    with torch.no_grad():
        result = model.forward_decoding(
            prompts_token_ids, valid_length, max_decode_len, 
            sampling_topk=sampling_topk, ignore_token_ids=(0, 1, 2),
        )  # ignore_token_ids: 0: padding, 1: unk, 2: sos
    
    print("-==Decoding samples==-")
    res = translate_logits(result, vocab.idx2str, vocab.unk, "<eos>")
    
    all_results = []
    
    for r in res:
        all_results.append("".join(r))
        print(all_results[-1])

    return all_results


class Sampler(object):
    
    def __init__(
        self, max_decode_len, sampling_topk, device, wv_path, model=None, 
        model_path=None, **kwargs,
    ):
        """Sampler for decoding.

        Args:
            max_decode_len ([type]): [description]
            sampling_topk ([type]): [description]
            device ([type]): [description]
            model ([type], optional): [description]. Defaults to None.
            
            if model is None, kwargs must be specified with following keys:
                num_embeddings
                embedding_dim
                hidden_dim
                rnn_layers
        """
        self.device = device
        self.vocab = Vocab(wv_path)
        self.max_decode_len = max_decode_len
        self.sampling_topk = sampling_topk

        if model is not None:
            self.model = model
        else:
            assert kwargs, "Model init params {num_embeddings, embeding_dim, "
            "hidden_dim, rnn_layers} must be specified!"

            self.model = Generator(
                **kwargs,
                device=self.device,
                pretrained_embeddings=None,
            )
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")["model"],
            )

    def sample(self, prompts):
        return do_sample(
            self.model, self.vocab, self.vocab.str2idx, prompts, 
            self.max_decode_len, self.sampling_topk, self.device,
        )
