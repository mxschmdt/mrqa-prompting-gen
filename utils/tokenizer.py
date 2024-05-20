from collections import defaultdict

from openprompt.plms import T5TokenizerWrapper


class MT5TokenizerWrapper(T5TokenizerWrapper):
    r"""
    mT5 tokenizer does not add the sentinel tokens to `additional_tokens_ids` hence we have to take them from the vocabulary directly
    """

    def mask_token(self, i):
        return f"<extra_id_{i}>"

    def mask_token_ids(self, i):
        return self.tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")


class BartTokenizerWrapper(T5TokenizerWrapper):
    r"""
    Bart tokenizer works as seq-to-seq model T5 but does not index masks
    """

    def mask_token(self, i):
        assert i == 0, "Bart can only handle single mask inputs"
        return f"<mask>"

    def mask_token_ids(self, i):
        assert i == 0, "Bart can only handle single mask inputs"
        return self.tokenizer.convert_tokens_to_ids(f"<mask>")

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        wrapped_example, others = wrapped_example

        if teacher_forcing:
            tgt_text = others["tgt_text"]
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0

        decoder_input_ids = []
        loss_ids = []

        for piece_id, piece in enumerate(wrapped_example):
            if piece["text"] == self.template_mask_token:
                assert num_mask_token_used == 0, "Bart only supports single mask inputs"
                if teacher_forcing:
                    loss_ids.append(0)
                    encode_text = [self.mask_token_ids(num_mask_token_used)]
                    tgt_text_ids = self.tokenizer.encode(
                        " " + tgt_text[num_mask_token_used], add_special_tokens=False
                    )
                    decoder_input_ids.extend(tgt_text_ids)
                    loss_ids.extend([1] * len(tgt_text_ids))
                else:
                    encode_text = [self.mask_token_ids(num_mask_token_used)]
                num_mask_token_used += 1
            else:
                if piece["text"] in self.special_tokens_maps.keys():
                    to_replace = self.special_tokens_maps[piece["text"]]
                    if to_replace is not None:
                        piece["text"] = to_replace
                    else:
                        raise KeyError(
                            "This tokenizer doesn't specify {} token.".format(
                                piece["text"]
                            )
                        )

                if "soft_token_ids" in piece and piece["soft_token_ids"] != 0:
                    encode_text = [
                        0
                    ]  # can be replace by any token, since these token will use their own embeddings
                else:
                    encode_text = self.tokenizer.encode(
                        piece["text"], add_special_tokens=False
                    )

            encoding_length = len(encode_text)

            encoder_inputs["input_ids"].append(encode_text)
            for key in piece:
                if key not in ["text", "loss_ids"]:
                    encoder_inputs[key].append([piece[key]] * encoding_length)

        # decoder input ids
        decoder_inputs = {"decoder_input_ids": decoder_input_ids, "loss_ids": loss_ids}
        decoder_inputs = self.truncate_decoder_inputs(decoder_inputs)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)

        # create special input ids
        encoder_inputs["attention_mask"] = [1] * len(encoder_inputs["input_ids"])
        # padding
        encoder_inputs = self.padding(
            input_dict=encoder_inputs,
            max_len=self.max_seq_length,
            pad_id_for_inputs=self.tokenizer.pad_token_id,
        )

        all_input_ids = {**encoder_inputs, **decoder_inputs}
        return all_input_ids
