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

# SentencePiece
import sentencepiece as spm

# Other
import sys
import glob
import os
import pdb
from speechbrain.dataio.encoder import CTCTextEncoder

def collate_fn_pad(batch):

    # Regular Mode, multilingual or monolingual
    if len(batch[0]) > 1:

        # Sorting sequences by lengths
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

        # Pad data sequences
        data = [item[0].squeeze() for item in sorted_batch]
        data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

        # Pad labels
        target = [item[1] for item in sorted_batch]
        target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

        if len(batch[0]) == 3:
            # multilingual
            ## language, int
            langs = torch.tensor([item[2] for item in sorted_batch])
            # langs = langs.unsqueeze(1).expand((-1, data.shape[1])).unsqueeze(2)
            # ## turn int into one-hot
            # one_hots = torch.zeros(langs.size(0), langs.size(1), 5).zero_()
            # langs = one_hots.scatter_(2, langs.data, 1)
            return data, target, data_lengths, target_lengths, langs
        elif len(batch[0]) == 2:
            # monolingual
            return data, target, data_lengths, target_lengths, None

    # LM Mode
    elif len(batch[0]) == 1:

        # Sort Batch
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        sorted_batch = [item[0] for item in sorted_batch]

        # Create Labels
        x = torch.nn.utils.rnn.pad_sequence(sorted_batch, batch_first=True, padding_value=0)
        x_len = torch.tensor([t.size(0) for t in sorted_batch], dtype=torch.long)
        y = [torch.cat([item, item.new_zeros(1)]) for item in sorted_batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)

        return x, x_len, y

    else:

        raise Exception("Batch Format Error")

def create_tokenizer(training_params, tokenizer_params):

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "LibriSpeech":

        # Corpus File Path
        corpus_path = training_params["training_dataset_path"] + training_params["training_dataset"] + "_corpus.txt"

        # Create Corpus File
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            corpus_file = open(corpus_path, "w")
            for file_path in glob.glob(training_params["training_dataset_path"] + "*/*/*/*.txt"):
                for line in open(file_path, "r").readlines():
                    corpus_file.write(line[len(line.split()[0]) + 1:-1].lower() + "\n")

        # Train Tokenizer
        print("Training Tokenizer")
        if tokenizer_params["vocab_type"] == "bpe":
            spm.SentencePieceTrainer.train(input=training_params["training_dataset_path"] + training_params["training_dataset"] + "_corpus.txt", model_prefix=tokenizer_params["tokenizer_path"].split(".model")[0], vocab_size=tokenizer_params["vocab_size"], character_coverage=1.0, model_type=tokenizer_params["vocab_type"], bos_id=-1, eos_id=-1, unk_surface="")
        elif tokenizer_params["vocab_type"] == "char":
            tokenizer = CTCTextEncoder()
            special_labels = {
                "blank_label": tokenizer_params["blank_index"],
            }
            tokenizer.load_or_create(
                path=tokenizer_params["tokenizer_path"],
                from_iterables=[line.strip("\n") for line in open(corpus_path, "r").readlines()],
                output_key=None,
                special_labels=special_labels,
                sequence_input=True,
            )
        print("Training Done")

def prepare_dataset(training_params, tokenizer_params, tokenizer):

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "LibriSpeech":

        # Read corpus
        print("Reading Corpus")
        label_paths = []
        sentences = []
        for file_path in glob.glob(training_params["training_dataset_path"] + "*/*/*/*.txt"):
            for line in open(file_path, "r").readlines():
                label_paths.append(file_path.replace(file_path.split("/")[-1], "") + line.split()[0] + "." + tokenizer_params["vocab_type"] + "_" + str(tokenizer_params["vocab_size"]))
                sentences.append(line[len(line.split()[0]) + 1:-1].lower())

        # Save Labels and lengths
        print("Encoding sequences")
        for i, (sentence, label_path) in enumerate(zip(sentences, label_paths)):

            # Print
            sys.stdout.write("\r{}/{}".format(i, len(label_paths)))

            # Tokenize and Save label
            label = torch.LongTensor(tokenizer.encode(sentence))
            torch.save(label, label_path)

            # Save Audio length
            audio_length = torchaudio.load(label_path.split(".")[0] + ".flac")[0].size(1)
            torch.save(audio_length, label_path.split(".")[0] + ".flac_len")

            # Save Label length
            label_length = label.size(0)
            torch.save(label_length, label_path + "_len")


def prepare_dataset_char(training_params, tokenizer_params, tokenizer):

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "LibriSpeech":

        # Read corpus
        print("Reading Corpus")
        label_paths = []
        sentences = []
        for file_path in glob.glob(training_params["training_dataset_path"] + "*/*/*/*.txt"):
            for line in open(file_path, "r").readlines():
                label_paths.append(file_path.replace(file_path.split("/")[-1], "") + line.split()[0] + "." + tokenizer_params["vocab_type"])
                sentences.append(line[len(line.split()[0]) + 1:-1].lower())

        # Save Labels and lengths
        print("Encoding sequences")
        for i, (sentence, label_path) in enumerate(zip(sentences, label_paths)):

            # Print
            sys.stdout.write("\r{}/{}".format(i, len(label_paths)))

            # Tokenize and Save label
            label = torch.LongTensor(tokenizer.encode_sequence(sentence))
            torch.save(label, label_path)

            # Save Audio length
            audio_length = torchaudio.load(label_path.split(".")[0] + ".flac")[0].size(1)
            torch.save(audio_length, label_path.split(".")[0] + ".flac_len")

            # Save Label length
            label_length = label.size(0)
            torch.save(label_length, label_path + "_len")
