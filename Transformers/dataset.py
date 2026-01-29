import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # NOTE: DID CHANGES
        """
            # ❌ Wrong - passing a list
            self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], ...)

            # ✅ Correct - passing a string
            self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], ...)
        """
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)
        # Dont know what is this?

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # no. padding tokens require for each token
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # no. other than [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # Decoder sequences only need [SOS] and not [EOS]

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Create 3 tensors
        """
            1. for encoder input
            2. for decoder input
            3. output of decoder (Actual target column)
        """
        # Basically, we are concatinating all the tokens to form a complete sentence for training
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len 

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # Encoder mask is the mask to not involve the padded tokens into self attention -> (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # The words next to current word will be hided from the decoder -> (1, 1, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
