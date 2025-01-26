import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, load_from_disk
from datasets import Audio, DatasetDict, Dataset, concatenate_datasets
from label_maker import LabelMaker
import os
import re
from pathlib import Path

CAPTION_MIN_LENGTH = 100
CAPTION_MAX_LENGTH = 900
AUDIO_MIN_DURATION = 2.0
AUDIO_MAX_DURATION = 28.0
LABEL_MAX_LENGTH = 440


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

    def __init__(self, dataset_name: str, processor, train_split: float = 0.95, max_rows: int=0,
                 dataset_column_audio: str = "audio.mp3", dataset_column_metadata: str = "metadata.json", dataset_column_file_name: str = "segment_filename",
                 dataset_column_duration: str = "duration_ms", dataset_column_duration_scale: float = 1000.0,
                 with_emotion=False, with_caption=True, with_detailed_caption=False, with_transcription=False) -> None:
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.feature_extractor = self.processor.feature_extractor
        self.collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        self.column_audio = dataset_column_audio
        self.column_metadata = dataset_column_metadata
        self.column_file_name = dataset_column_file_name
        self.column_duration = dataset_column_duration
        self.column_duration_scale = dataset_column_duration_scale
        self.with_emotion = with_emotion
        self.with_caption = with_caption
        self.with_detailed_caption = with_detailed_caption
        self.with_transcription = with_transcription
        self.label_maker = LabelMaker()
        
        if Path(dataset_name).exists():
            self.dataset = load_from_disk(dataset_name)
            self.dataset["val"] = self.dataset["validation"]
        else:
            self.dataset = load_dataset(dataset_name)
            if max_rows > 0:
                self.dataset['train'] = self.dataset['train'].select(range(max_rows))
            self.dataset = self.dataset.cast_column(dataset_column_audio, Audio(sampling_rate=16000))
            train_split = max(min(train_split, 0.9999), 0.7)
            self.dataset = self.dataset['train'].train_test_split(test_size=1-train_split, shuffle=True, seed=42)
            ds = self.dataset['test'].train_test_split(test_size=0.5, shuffle=True, seed=42)
            self.dataset["validation"] = ds["train"]
            self.dataset["test"] = ds["test"]
            # num_proc = max(8, int(len(os.sched_getaffinity(0))/2))
            for split in self.dataset:
                # self.dataset[split] = self.dataset[split].filter(self.caption_length_check, num_proc=num_proc)
                self.dataset[split] = self.dataset[split].map(self.prepare_dataset, num_proc=8,
                                                            remove_columns=['__key__', '__url__'])
                self.dataset[split] = self.dataset[split].filter(self.row_check, num_proc=8)

            self.dataset["val"] = self.dataset["validation"]
            self.dataset["train_mini"] = self.dataset["train"].select(range(8))
            self.dataset["val_mini"] = self.dataset["val"].select(range(32))
        # print(self.dataset)

    def get_dataset(self):
        return self.dataset
    
    def get_collator(self):
        return self.collator

    def prepare_label(self, caption: str):
        forced_ac_decoder_ids = self.tokenizer("", text_target=self.prefix, add_special_tokens=False).labels
        *fluff_tokens, eos = self.tokenizer("", text_target="", add_special_tokens=True).labels
        labels = self.tokenizer("", text_target=caption, add_special_tokens=False).labels
        labels = fluff_tokens + forced_ac_decoder_ids + labels+ [eos]
        return labels, forced_ac_decoder_ids
    
    def prepare_dataset(self, batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch[self.column_audio]

        # create label from metadata
        label = self.label_maker.create_label(batch[self.column_metadata],
                                              with_emotion=self.with_emotion, with_caption=self.with_caption,
                                              with_detailed_caption=self.with_detailed_caption, with_transcription=self.with_transcription)

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        
        # encode target text to label ids
        # batch["labels"] = self.tokenizer(transcription).input_ids
        batch["labels"], batch["forced_ac_decoder_ids"] = self.prepare_label(label)
        return batch
    
    def get_val_alternatives(self):
        key = ('laion', 'caption')
        values = {}
        for row in self.dataset["val"]:
            label = self.label_maker.create_label(row[self.column_metadata],
                                                  with_emotion=self.with_emotion, with_caption=self.with_caption,
                                                  with_detailed_caption=self.with_detailed_caption, with_transcription=self.with_transcription)
            values[label] = [label]
        val_alternatives = {
            key: values
        }
        return val_alternatives
    
    def caption_length_check(self, row):
        if self.column_duration in row[self.column_metadata] \
                and (row[self.column_metadata][self.column_duration] < AUDIO_MIN_DURATION*self.column_duration_scale or \
                    row[self.column_metadata][self.column_duration] > AUDIO_MAX_DURATION*self.column_duration_scale):
            return False
        elif len(row[self.column_metadata]['caption']) > CAPTION_MAX_LENGTH:
            return False
        elif len(row[self.column_metadata]['caption']) >= CAPTION_MIN_LENGTH:
            return True
        elif 'transcription' in row[self.column_metadata]:
            if row[self.column_metadata]['transcription'] is not None \
                and len(row[self.column_metadata]['transcription']) >= CAPTION_MIN_LENGTH \
                and len(row[self.column_metadata]['transcription']) <= CAPTION_MAX_LENGTH :
                return True
            else:
                return False
        else:
            return False
    

    def label_length_check(self, row):
        if len(row['labels']) < LABEL_MAX_LENGTH:
            return True
        else:
            return False


    def row_check(self, row):
        if self.column_duration in row[self.column_metadata] \
                and (row[self.column_metadata][self.column_duration] < AUDIO_MIN_DURATION*self.column_duration_scale or \
                    row[self.column_metadata][self.column_duration] > AUDIO_MAX_DURATION*self.column_duration_scale):
            return False
        if 20 < len(row['labels']) < LABEL_MAX_LENGTH:
            return True
        else:
            return False