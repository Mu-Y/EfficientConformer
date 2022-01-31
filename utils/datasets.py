# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torchaudio

# Sentencepiece
import sentencepiece as spm

# Other
import glob
from tqdm import tqdm
import csv
import pdb

# Librispeech 292.367 samples
class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_paths, training_params, tokenizer_params, args, filter_length=True):

        self.max_seconds_per_dataset = training_params["max_hours_per_dataset"] * 3600
        ## load data from csv file
        data = {}
        for csv_path in csv_paths:
            total_duration = 0.
            with open(csv_path, newline="") as csvfile:
                reader = csv.DictReader(csvfile, skipinitialspace=True)
                for row in reader:
                    data_id = row["ID"]
                    if data_id in data:
                        raise ValueError(f"Duplicate id: {data_id}")
                    if "duration" in row:
                        row["duration"] = float(row["duration"])
                        if filter_length:
                            if row["duration"] > training_params["max_seconds_per_utt"]:
                                continue
                    if "wav_len" in row:
                        row["wav_len"] = int(row["wav_len"])
                    if "label_len" in row:
                        row["label_len"] = int(row["label_len"])
                    data[data_id] = row
                    total_duration += row["duration"]
                    if total_duration > self.max_seconds_per_dataset:
                        break
            print("{}: total hours: {}".format(csv_path, total_duration/3600))
        self.data = [d for _, d in data.items()]

        self.vocab_type = tokenizer_params["vocab_type"]
        if tokenizer_params["vocab_type"] == "bpe":
            self.vocab_size = str(tokenizer_params["vocab_size"])
        else:
            self.vocab_size = None
        self.lm_mode = training_params.get("lm_mode", False)

        # if split.split("-")[0] == "train":
        #     self.names = self.filter_lengths(training_params["train_audio_max_length"], training_params["train_label_max_length"], args.rank)
        # else:
        #     self.names = self.filter_lengths(training_params["eval_audio_max_length"], training_params["eval_audio_max_length"], args.rank)
        # if filter_length:
        #     self.data = self.filter_lengths(training_params["train_audio_max_length"],
        #                                     training_params["train_label_max_length"],
        #                                     args.rank)
        if training_params["sort_dataset"] == "ascending":
            self.data = sorted(self.data, key=lambda x:x["duration"], reverse=False)
        elif training_params["sort_dataset"] == "descending":
            self.data = sorted(self.data, key=lambda x:x["duration"], reverse=True)

        if training_params["multilingual"]:
            self.lang2idx = training_params["lang2idx"]
        else:
            self.lang2idx = None



    def __getitem__(self, i):

        if self.lm_mode:
            return [torch.load(self.names[i].split(".flac")[0].split("_")[0] + "." + self.vocab_type + "_" + self.vocab_size)]
        else:
            if self.lang2idx:
                return [torchaudio.load(self.data[i]["wav_path"])[0],
                        torch.load(self.data[i]["tokenized_path"]),
                        self.lang2idx[self.data[i]["lang"]]]
            else:
                return [torchaudio.load(self.data[i]["wav_path"])[0],
                        torch.load(self.data[i]["tokenized_path"])]

    def __len__(self):

        return len(self.data)

    def filter_lengths(self, audio_max_length, label_max_length, rank=0):

        if audio_max_length is None or label_max_length is None:
            return self.data

        if rank == 0:
            print("dataset filtering")
            print("Audio maximum length : {} / Label sequence maximum length : {}".format(audio_max_length, label_max_length))
            self.data = tqdm(self.data)

        return [d for d in self.data if d["wav_len"] <= audio_max_length and d["label_len"] <= label_max_length]

# Librispeech Corpus 40.418.261 samples
class LibriSpeechCorpusDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, training_params, tokenizer_params, split, args):

        # Dataset Params
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_params["tokenizer_path"])
        self.corpus = open(dataset_path, 'r').readlines()
        self.max_len = training_params["train_label_max_length"]

    def __getitem__(self, i):

        if self.max_len:
            while len(self.tokenizer.encode(self.corpus[i][:-1].lower())) > self.max_len:
                i = torch.randint(0, self.__len__(), [])

        return [torch.LongTensor(self.tokenizer.encode(self.corpus[i][:-1].lower()))]

    def __len__(self):

        return len(self.corpus)
