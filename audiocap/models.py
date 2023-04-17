import transformers
import torch
from typing import Optional, List, Tuple, Union
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.nn import CrossEntropyLoss
from transformers.generation.logits_process import WhisperTimeStampLogitsProcessor
from transformers import WhisperPreTrainedModel, WhisperForConditionalGeneration


# the only allowed language and task
LANGUAGES = {
    "en": "english"}

TO_LANGUAGE_CODE = {
    "english": "en"
}

TASK_IDS = ["transcribe"]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class WhisperForAudioCaptioning(WhisperForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        forced_ac_decoder_ids: Optional[torch.Tensor] = None,  # only added not to throw errors when seen
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset
        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> generated_ids = model.generate(inputs=input_features)
        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            forced_ac_decoder_ids: torch.Tensor = None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            return_timestamps=None,
            task="transcribe",
            language="english",
            **kwargs,
        ):
            """
            Generates sequences of token ids for models with a language modeling head.
            <Tip warning={true}>
            Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
            model's default generation configuration. You can override any `generation_config` by passing the corresponding
            parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.
            For an overview of generation strategies and code examples, check out the [following
            guide](./generation_strategies).
            </Tip>
            Parameters:
                inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                    The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                    method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                    should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                    `input_ids`, `input_values`, `input_features`, or `pixel_values`.
                generation_config (`~generation.GenerationConfig`, *optional*):
                    The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                    passed to generate matching the attributes of `generation_config` will override them. If
                    `generation_config` is not provided, the default will be used, which had the following loading
                    priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                    configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                    default values, whose documentation should be checked to parameterize generation.
                logits_processor (`LogitsProcessorList`, *optional*):
                    Custom logits processors that complement the default logits processors built from arguments and
                    generation config. If a logit processor is passed that is already created with the arguments or a
                    generation config an error is thrown. This feature is intended for advanced users.
                stopping_criteria (`StoppingCriteriaList`, *optional*):
                    Custom stopping criteria that complement the default stopping criteria built from arguments and a
                    generation config. If a stopping criteria is passed that is already created with the arguments or a
                    generation config an error is thrown. This feature is intended for advanced users.
                prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                    If provided, this function constraints the beam search to allowed tokens only at each step. If not
                    provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                    `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                    on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                    for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                    Retrieval](https://arxiv.org/abs/2010.00904).
                synced_gpus (`bool`, *optional*, defaults to `False`):
                    Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
                return_timestamps (`bool`, *optional*):
                    Whether to return the timestamps with the text. This enables the `WhisperTimestampsLogitsProcessor`.
                task (`bool`, *optional*):
                    Task to use for generation, either "translate" or "transcribe". The `model.config.forced_decoder_ids`
                    will be updated accordingly.
                language (`bool`, *optional*):
                    Language token to use for generation, can be either in the form of `<|en|>`, `en` or `english`. You can
                    find all the possible language tokens in the `model.generation_config.lang_to_id` dictionary.
                kwargs:
                    Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                    forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                    specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.
            Return:
                [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
                    If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                    [`~utils.ModelOutput`] types are:
                        - [`~generation.GreedySearchDecoderOnlyOutput`],
                        - [`~generation.SampleDecoderOnlyOutput`],
                        - [`~generation.BeamSearchDecoderOnlyOutput`],
                        - [`~generation.BeamSampleDecoderOnlyOutput`]
                    If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                    [`~utils.ModelOutput`] types are:
                        - [`~generation.GreedySearchEncoderDecoderOutput`],
                        - [`~generation.SampleEncoderDecoderOutput`],
                        - [`~generation.BeamSearchEncoderDecoderOutput`],
                        - [`~generation.BeamSampleEncoderDecoderOutput`]
            """
            if generation_config is None:
                generation_config = self.generation_config

            if return_timestamps is not None:
                if not hasattr(generation_config, "no_timestamps_token_id"):
                    raise ValueError(
                        "You are trying to return timestamps, but the generation config is not properly set."
                        "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`."
                        "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    )

                generation_config.return_timestamps = return_timestamps
            else:
                generation_config.return_timestamps = False

            if language is not None:
                generation_config.language = language
            if task is not None:
                generation_config.task = task

            forced_decoder_ids = []
            if task is not None or language is not None:
                if hasattr(generation_config, "language"):
                    if generation_config.language in generation_config.lang_to_id.keys():
                        language_token = generation_config.language
                    elif generation_config.language in TO_LANGUAGE_CODE.keys():
                        language_token = f"<|{TO_LANGUAGE_CODE[generation_config.language]}|>"
                    else:
                        raise ValueError(
                            f"Unsupported language: {language}. Language should be one of:"
                            f" {list(TO_LANGUAGE_CODE.keys()) if generation_config.language in TO_LANGUAGE_CODE.keys() else list(TO_LANGUAGE_CODE.values())}."
                        )
                    forced_decoder_ids.append((1, generation_config.lang_to_id[language_token]))
                else:
                    forced_decoder_ids.append((1, None))  # automatically detect the language

                if hasattr(generation_config, "task"):
                    if generation_config.task in TASK_IDS:
                        forced_decoder_ids.append((2, generation_config.task_to_id[generation_config.task]))
                    else:
                        raise ValueError(
                            f"The `{generation_config.task}`task is not supported. The task should be one of `{TASK_IDS}`"
                        )
                else:
                    forced_decoder_ids.append((2, generation_config.task_to_id["transcribe"]))  # defaults to transcribe
                if hasattr(generation_config, "no_timestamps_token_id") and not generation_config.return_timestamps:
                    idx = forced_decoder_ids[-1][0] + 1 if forced_decoder_ids else 1
                    forced_decoder_ids.append((idx, generation_config.no_timestamps_token_id))

            # Legacy code for backward compatibility
            elif hasattr(self.config, "forced_decoder_ids") and self.config.forced_decoder_ids is not None:
                forced_decoder_ids = self.config.forced_decoder_ids
            elif (
                hasattr(self.generation_config, "forced_decoder_ids")
                and self.generation_config.forced_decoder_ids is not None
            ):
                forced_decoder_ids = self.generation_config.forced_decoder_ids

            if generation_config.return_timestamps:
                logits_processor = [WhisperTimeStampLogitsProcessor(generation_config)]

            if len(forced_decoder_ids) > 0:
                # get the token sequence coded in forced_decoder_ids
                forced_decoder_ids_token_ids = [tok_id for _, tok_id in sorted(forced_decoder_ids, key=lambda x: x[0])]

                # enrich every sample's forced_ac_decoder_ids with Whisper's forced_decoder_ids
                expanded_forced_decoder_ids_token_ids = torch.tensor(forced_decoder_ids_token_ids).expand((forced_ac_decoder_ids.size()[0],
                                                                                                          len(forced_decoder_ids_token_ids)))
                device = "cuda" if forced_ac_decoder_ids.get_device() == 0 else "cpu"
                expanded_forced_decoder_ids_token_ids = expanded_forced_decoder_ids_token_ids.to(device)
                decoder_input_ids = torch.cat((expanded_forced_decoder_ids_token_ids, forced_ac_decoder_ids), dim=1)

                generation_config.forced_decoder_ids = forced_decoder_ids

            return super(WhisperPreTrainedModel, self).generate(   # changed by adam (calling grandparent)
                inputs,
                generation_config,
                logits_processor,
                stopping_criteria,
                prefix_allowed_tokens_fn,
                synced_gpus,
                decoder_input_ids=decoder_input_ids,
                **kwargs,
            )