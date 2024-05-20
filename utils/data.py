from dataclasses import dataclass
from typing import List, Union

from transformers import PreTrainedTokenizer


@dataclass
class T5Postprocessor:
    tokenizer: PreTrainedTokenizer
    num_masks: int = 1

    def __call__(
        self, sequences: List[int], input_lengths, num_masks: Union[None, int] = None
    ) -> List[int]:
        if num_masks is None:
            num_masks = self.num_masks
        generated_sentences = []
        if isinstance(input_lengths, int):
            input_lengths = [input_lengths] * len(sequences)
        for sent_id, seq in enumerate(sequences):
            generated_sentences.append([])
            seq = seq[input_lengths[sent_id] :]
            seq_decoded = self.tokenizer.decode(
                seq, clean_up_tokenization_spaces=True, skip_special_tokens=False
            )

            for mask_idx in range(1, num_masks + 1):
                mask_token = self.mask_token(mask_idx)
                idx = seq_decoded.find(mask_token) if mask_token is not None else -1
                if idx >= 0:
                    cur_seq_decoded = seq_decoded[:idx]
                    seq_decoded = seq_decoded[idx + len(mask_token) :]
                else:
                    cur_seq_decoded = seq_decoded
                if mask_idx < num_masks:
                    generated_sentences[-1].append(cur_seq_decoded.strip())

            if (
                hasattr(self.tokenizer, "eos_token")
                and self.tokenizer.eos_token is not None
            ):
                idx = cur_seq_decoded.find(self.tokenizer.eos_token)
                if idx >= 0:
                    cur_seq_decoded = cur_seq_decoded[:idx]
            generated_sentences[-1].append(cur_seq_decoded.strip())
        return generated_sentences

    def mask_token(self, i: int):
        return (
            self.tokenizer.additional_special_tokens[i]
            if i < len(self.tokenizer.additional_special_tokens)
            else None
        )


class MT5Postprocessor(T5Postprocessor):
    def mask_token(self, i: int):
        # mT5 tokenizer does not add the sentinel tokens to a separate list hence we have to convert them the ordinary way
        return f"<extra_id_{i}>" if 0 <= i <= 99 else None


POSTPROCESSOR_CLASSES = {
    "t5": T5Postprocessor,
    "mt5": MT5Postprocessor,
}
