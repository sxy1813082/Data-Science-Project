import torch
import os
import numpy as np
import argparse

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader

# Passing arguments in via the command line --------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained", 
    type=str, 
    required=True, 
    help="the pre-trained model to run"
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    default="noexp",
    help="Model type: noexp, expbert, bertie",
)
parser.add_argument(
    "--subset",
    type=str,
    required=False,
    default="all",
    help="what batch we are on e.g. subset_1",
)

args = parser.parse_args()


def main():
    # initialises the model depending on which method is being used
    if args.model == "bertie":
        model = AutoModelForSequenceClassification.from_pretrained(args.pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    else:
        model = AutoModel.from_pretrained(args.pretrained)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

    # loads in either the original dataset or the expanded dataset
    if args.model == "noexp":
        temp_path = "./data/"
        raw_dataset = load_from_disk(temp_path)
    else:
        temp_path = "./data/exp/" + args.subset
        raw_dataset = load_from_disk(temp_path)

    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    # takes in only the tweet and creates an embedding
    def tokenize_noexp_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=True, return_tensors="pt"
        )

    # takes in both the tweet and the explanation and creates an embedding
    def tokenize_exp_function(examples):
        return tokenizer(
            examples["text"],
            examples["exp_and_td"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

    # optimises the runtime if running on a GPU
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    torch.cuda.empty_cache()

    # splits the dataset into batches of size 10 and passes them through the tokenizer and pre-trained model.
    emb = []
    train_dataloader = DataLoader(raw_dataset["train"], batch_size=10)
    if args.model == "bertie":
        model.eval()
        for batch in train_dataloader:
            with torch.no_grad():
                tokenized_train = tokenize_exp_function(batch)
                model_outputs = model(**tokenized_train)
                embeddings = model_outputs["logits"]
                embeddings = embeddings.cpu().detach().numpy()
                emb.append(embeddings)
                torch.cuda.empty_cache()

    elif args.model == "expbert":
        for batch in train_dataloader:
            with torch.no_grad():
                tokenized_train = tokenize_exp_function(batch)
                model_outputs = model(**tokenized_train)
                output = model_outputs["last_hidden_state"]
                embeddings = output[:, 0, :]
                embeddings = embeddings.cpu().detach().numpy()
                emb.append(embeddings)
                torch.cuda.empty_cache()

    else:
        for batch in train_dataloader:
            with torch.no_grad():
                tokenized_train = tokenize_noexp_function(batch)
                model_outputs = model(**tokenized_train)
                output = model_outputs["last_hidden_state"]
                embeddings = output[:, 0, :]
                embeddings = embeddings.cpu().detach().numpy()
                emb.append(embeddings)
                torch.cuda.empty_cache()

    # converts the embeddings into a tensor and reshapes them to the correct size
    emb = np.array(emb)
    emb = np.vstack(emb)

    embeddings = torch.tensor(emb)
    print(embeddings.shape)

    if args.model == "bertie":
        embeddings = torch.reshape(embeddings, (2000, 36 * 3))
        print(embeddings.shape)
    elif args.model == "expbert":
        embeddings = torch.reshape(embeddings, (2000, 36 * 768))
        print(embeddings.shape)
    else:
        print(embeddings.shape)

    # creates a filename using the passed in arguments
    # and then saves the embedding with this name
    save_filename = (
        "./embeddings/NEW_"
        + args.model
        + "_embeddings_"
        + args.pretrained
        + "_"
        + args.subset
        + ".pt"
    )
    print(save_filename)
    torch.save(embeddings, save_filename)


if __name__ == "__main__":
    main()
