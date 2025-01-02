import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch_forced_ac_decoder_ids = [feature["forced_ac_decoder_ids"] for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["forced_ac_decoder_ids"] = torch.tensor(batch_forced_ac_decoder_ids)

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
    

class DataLaion():
    prefix = "laion > caption: "

    def __init__(self, dataset_name: str, processor, train_split: float = 0.9) -> None:
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        self.dataset = load_dataset(dataset_name)
        # self.dataset['train'] = self.dataset['train'].select(range(100))
        train_split = max(min(train_split, 0.99), 0.7)
        self.dataset = self.dataset['train'].train_test_split(test_size=1-train_split, shuffle=True, seed=42)
        ds = self.dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
        self.dataset["validation"] = ds["train"]
        self.dataset["test"] = ds["test"]
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].map(self.prepare_dataset, num_proc=8)
        self.dataset["val"] = self.dataset["validation"]
        self.dataset["train_mini"] = self.dataset["train"].select(range(8))
        self.dataset["val_mini"] = self.dataset["val"].select(range(32))

    def get_dataset(self):
        return self.dataset
    
    def get_collator(self):
        return self.collator

    def prepare_label(self, caption: str):
        forced_ac_decoder_ids = self.tokenizer("", text_target=self.prefix, add_special_tokens=False).labels
        *fluff_tokens, eos = self.tokenizer("", text_target="", add_special_tokens=True).labels
        labels = self.tokenizer("", text_target=caption, add_special_tokens=False).labels
        labels = fluff_tokens + forced_ac_decoder_ids + labels + [eos]
        return labels, forced_ac_decoder_ids
    
    def prepare_dataset(self, batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio.mp3"]

        # optional pre-processing steps
        caption = batch['metadata.json']['caption']
        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        # encode target text to label ids
        # batch["labels"] = self.tokenizer(transcription).input_ids
        batch["labels"], batch["forced_ac_decoder_ids"] = self.prepare_label(caption)
        return batch
    
    def get_val_alternatives(self):
        key = ('laion', 'caption')
        values = {}
        for row in self.dataset["val"]:
            cap = row["metadata.json"]["caption"]
            values[cap] = [cap]
        val_alternatives = {
            key: values
        }
        return val_alternatives
