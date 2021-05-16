import logging
import torch
from torch.nn.utils import clip_grad_norm_

from nlkit.trainer import BaseTrainer
from nlkit.utils import Phase, check_should_do_early_stopping

from tqdm import tqdm

from .utils import translate_logits

logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):

    def __init__(
        self, model, train_data_loader, valid_data_loader, test_data_loader, 
        lr_scheduler, optimizer, weight_init, summary_path, device, criterion,
        total_epoch, model_path, gradient_clip, not_early_stopping_at_first,
        es_with_no_improvement_after, verbose, vocab, max_decode_len,
        idx2str, str2idx,
    ):
        
        super(Trainer, self).__init__(
            model, train_data_loader, valid_data_loader, test_data_loader, 
            lr_scheduler, optimizer, weight_init, summary_path, device, 
            criterion, total_epoch, model_path
        )
        
        self.gradient_clip = gradient_clip
        self.not_early_stopping_at_first = not_early_stopping_at_first
        self.es_with_no_improvement_after = es_with_no_improvement_after
        self.verbose = verbose
        self.vocab = vocab
        self.max_decode_len = max_decode_len
        
        self.loss_record_on_valid = []
        self.train_record = []
        self.idx2str = idx2str
        self.str2idx = str2idx
        
    def iteration(self, epoch, data_loader, phase):
        data_iter = tqdm(
            enumerate(data_loader),
            desc="EP:{}:{}".format(phase.name, epoch),
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}",
        )

        total_loss = []

        for idx, data in data_iter:
            
            if phase == Phase.TRAIN:
                self.global_train_step += 1
            elif phase == Phase.VALID:
                self.global_valid_step += 1
            else:
                self.global_test_step += 1

            # data to device
            data = {key: value.to(self.device) for key, value in data.items()}

            # forward the model
            if phase == Phase.TRAIN:
                loss = self.forward_model(data)
                
            else:
                with torch.no_grad():
                    loss = self.forward_model(data)
                    
            total_loss.append(loss.item())

            # do backward if on train
            if phase == Phase.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()

                if self.gradient_clip:
                    clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip,
                    )
                    
                self.optimizer.step()

            log_info = {
                "phase": phase.name,
                "epoch": epoch,
                "iter": idx,
                "curr_loss": loss.item(),
                "avg_loss": sum(total_loss) / len(total_loss),
            }

            if self.verbose and not idx % self.verbose:
                data_iter.write(str(log_info))

        if phase == Phase.TRAIN:
            self.lr_scheduler.step()  # step every train epoch

        avg_loss = sum(total_loss) / len(total_loss)
        
        logger.info(
            "EP:{}_{}, avg_loss={}".format(
                epoch,
                phase.name,
                avg_loss,
            ),
        )

        # 记录训练信息
        record = {
            "epoch": epoch,
            "status": phase.name,
            "avg_loss": avg_loss,
        }

        self.train_record.append(record)

        # check should early stopping at valid
        if phase == Phase.VALID:
            self.loss_record_on_valid.append(avg_loss)

            should_stop = check_should_do_early_stopping(
                self.loss_record_on_valid,
                self.not_early_stopping_at_first,
                self.es_with_no_improvement_after,
                acc_like=False,
            )

            if should_stop:
                best_epoch = should_stop
                logger.info("Now stop training..")
                return best_epoch

        self.forward_sampling(
            prompts=[
                ["<sos>", "梦", "里"], 
                ["<sos>", "花", "开"], 
                ["<sos>", "大", "漠"],
            ],
        )
        return False
        
    def forward_model(self, data):
        loss = self.model.forward_loss(
            data["src"], data["tgt"], data["valid_length"],
        )
        return loss

    def forward_sampling(self, prompts):
        self.model.eval()
        prompts_ids = [
            [self.vocab.str2idx.get(i, self.str2idx["<unk>"]) for i in prompt]
            for prompt in prompts
        ]
        valid_length = torch.tensor([len(i) for i in prompts_ids])

        prompts_token_ids = torch.tensor(prompts_ids).to(self.device)
        
        with torch.no_grad():
            result = self.model.forward_decoding(
                prompts_token_ids, valid_length, self.max_decode_len,
            )
        
        print("-==Decoding samples==-")
        res = translate_logits(
            result, self.vocab.idx2str, self.vocab.unk, "<eos>"
        )
        
        all_results = []
        
        for r in res:
            all_results.append("".join(r))
            print(all_results[-1])

        return all_results
