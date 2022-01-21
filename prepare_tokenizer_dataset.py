# PyTorch
import torch
import torchaudio

# Other
import sys
import glob
import os
import pdb
from speechbrain.dataio.encoder import CTCTextEncoder
import argparse
import json
import csv

def create_tokenizer(training_params, tokenizer_params, corpus_path=None):

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


    elif training_params["training_dataset"] == "MLS+LibriSpeech":
        assert isinstance(training_params["mls_dataset_path"], list)
        assert training_params["librispeech_dataset_path"]
        assert corpus_path, print("For MLS + Librispeech case, you have to indicate a corpus_path to store the text corpus to train a tokenizer.")
        # Create Corpus File
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            corpus_file = open(corpus_path, "w")
            # librispeech
            for file_path in glob.glob(training_params["librispeech_dataset_path"] + "train*/*/*/*.txt"):
                for line in open(file_path, "r").readlines():
                    corpus_file.write(line[len(line.split()[0]) + 1:-1].lower() + "\n")
            # mls dataset
            for d_path in training_params["mls_dataset_path"]:
                for line in open(os.path.join(d_path, "train/transcripts.txt"), "r").readlines():
                    corpus_file.write(line.split("\t")[1].strip().lower() + "\n")


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

    return tokenizer

def prepare_dataset_char(training_params, tokenizer_params, tokenizer):

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "LibriSpeech":

        # Read corpus
        print("Reading Corpus")
        splits = ["train-clean-100",
                  "train-clean-360",
                  "train-other-500",
                  "dev-clean",
                  "dev-other",
                  "test-clean",
                  "test-other"]
        for split in splits:
            wav_paths = []
            sentences = []
            for file_path in glob.glob(training_params["training_dataset_path"] + \
                                       "{}/*/*/*.txt".format(split)):
                for line in open(file_path, "r").readlines():
                    wav_path = file_path.replace(file_path.split("/")[-1], "") + \
                                   line.split()[0] + ".flac"
                    sentence = line[len(line.split()[0]) + 1:-1].lower()
                    wav_paths.append(wav_path)
                    sentences.append(sentence)
            make_csv(training_params["training_dataset_path"],
                     wav_paths,
                     sentences,
                     split,
                     tokenizer_params,
                     tokenizer,
                     "english")
    # MLS dataset
    if training_params["training_dataset"] == "MLS+LibriSpeech":
        splits = ["train",
                  "dev",
                  "test"]
        for d_path in training_params["mls_dataset_path"]:
            lang = d_path.split("/")[-1].split("_")[1]
            print(lang)
            for split in splits:
                wav_paths = []
                sentences = []
                for line in open(os.path.join(d_path, split, "transcripts.txt"), "r").readlines():
                    wav_id = line.split("\t")[0]
                    wav_path = os.path.join(d_path, split, "audio",
                                            wav_id.split("_")[0], wav_id.split("_")[1], wav_id + ".flac")
                    assert os.path.isfile(wav_path)
                    sentence = line.split("\t")[1].strip().lower()
                    wav_paths.append(wav_path)
                    sentences.append(sentence)
                make_csv(d_path,
                         wav_paths,
                         sentences,
                         split,
                         tokenizer_params,
                         tokenizer,
                         lang)


def make_csv(save_folder, wav_lst, text_lst, split, tokenizer_params, tokenizer, lang):
    csv_file = os.path.join(save_folder, split + ".csv")
    csv_lines = [["ID", "lang", "duration", "wav_path", "wrd", "tokenized_path", "wav_len", "label_len"]]
    print("\nCreating csv for {}".format(split))
    for i, (wav_file, sentence) in enumerate(zip(wav_lst, text_lst)):
        # Print
        sys.stdout.write("\r{}/{}".format(i, len(wav_lst)))

        tokenized_path = wav_file.split(".")[0] + "." + tokenizer_params["vocab_type"]
        # Tokenize and Save label
        label = torch.LongTensor(tokenizer.encode_sequence(sentence))
        torch.save(label, tokenized_path)

        snt_id = wav_file.split("/")[-1].replace(".flac", "")
        # Audio length
        signal, fs = torchaudio.load(wav_file)
        duration = signal.shape[1] / fs
        wav_length = signal.shape[1]

        # Label length
        label_length = label.size(0)

        csv_line = [
            snt_id,
            lang,
            str(duration),
            wav_file,
            sentence,
            tokenized_path,
            wav_length,
            label_length
        ]
        csv_lines.append(csv_line)

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)


def main(args):

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)


    # Create Tokenizer
    if args.create_tokenizer:

        print("Creating Tokenizer")
        tokenizer = create_tokenizer(config["training_params"],
                                     config["tokenizer_params"],
                                     args.corpus_file)

    # Prepare Dataset
    if args.prepare_dataset:

        print("Preparing dataset")

        tokenizer = CTCTextEncoder()
        tokenizer.load(config["tokenizer_params"]["tokenizer_path"])
        if config["tokenizer_params"]["vocab_type"] == "bpe":
            prepare_dataset(config["training_params"], config["tokenizer_params"], tokenizer)
        elif config["tokenizer_params"]["vocab_type"] == "char":
            prepare_dataset_char(config["training_params"], config["tokenizer_params"], tokenizer)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--create_tokenizer", action="store_true")
    p.add_argument("--corpus_file", type=str, default=None, help="required when create_tokenizer is set")
    p.add_argument("--prepare_dataset", action="store_true")
    args = p.parse_args()

    main(args)
