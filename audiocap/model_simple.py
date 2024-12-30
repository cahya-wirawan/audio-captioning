import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    

class ModelSimple():
    def __init__(self) -> None:
        self.model_name = "openai/whisper-tiny"
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_name, language="en", task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(self.model_name, language="en", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.dataset = load_dataset("google/fleurs", "af_za")
        self.dataset.map(self.prepare_dataset, num_proc=8)
        self.dataset["val"] = self.dataset["validation"]

        self.collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        self.metric = evaluate.load("wer")

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset
    
    def get_tokenizer(self):
        return self.tokenizer

    def get_collator(self):
        return self.collator

    def get_compute_metrics(self):
        return self.compute_metrics

    def prepare_dataset(self, batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]

        # optional pre-processing steps
        transcription = batch["sentence"]
        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        # encode target text to label ids
        batch["labels"] = self.tokenizer(transcription).input_ids
        return batch
    
    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}