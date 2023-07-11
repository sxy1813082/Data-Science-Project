import os
import shutil

import pandas as pd
import emoji
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing
from torch import nn, multiprocessing


# creates expanded dataset (tweets and explanations)
def create_explanations_dataset(df, explanations):
    textual_descriptions = [
        "injured or dead people",
        "missing trapped or found people",
        "displaced people and evacuations",
        "infrastructure and utilities damage",
        "donation needs or offers or volunteering services",
        "caution and advice",
        "sympathy and emotional support",
        "other useful information",
        "not related or irrelevant",
    ]

    # concatenates the labels to the end of the explanations
    ex_td = explanations + textual_descriptions
    len_df = len(df.index)

    # creates N copies of 'ex_td' where N is the number of tweets
    df = df.iloc[np.repeat(np.arange(len(df)), len(ex_td))].reset_index(drop=True)
    ex_td = ex_td * len_df

    # adds each explanation and textual description to each tweet
    df.insert(1, "exp_and_td", ex_td, allow_duplicates=True)

    return df

def process_batch(batch, model, tokenizer, tokenize_exp_function):
    with torch.no_grad():
        tokenized_train = tokenize_exp_function(batch, tokenizer)
        model_outputs = model(**tokenized_train)
        embeddings = model_outputs["logits"]
        embeddings = embeddings.cpu().detach().numpy()
        return embeddings

# helper function - reads the explanations from a text file
def read_explanations(explanation_file):
    f = open(explanation_file, "r")
    lines = f.readlines()
    explanations = [line.strip() for line in lines]
    return explanations

def tokenize_exp_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        examples["exp_and_td"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )



def preprocess_samples_unxep(raw_dataset_noexp):
    # orgfile_path = "./embeddings/NEW_bertie_embeddings_textattack/" + "bert-base-uncased-MNLI_subset_1.pt"
    # orgembeddings = torch.load(orgfile_path)
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    if not os.path.exists("unexp_embeddings"):
        os.makedirs("unexp_embeddings")
    emb = []
    train_dataloader = DataLoader(raw_dataset_noexp["train"], batch_size=10)
    for batch in train_dataloader:
        with torch.no_grad():
            tokenized_train = tokenize_exp_function(batch,tokenizer)
            model_outputs = model(**tokenized_train)
            embeddings = model_outputs["logits"]
            embeddings = embeddings.cpu().detach().numpy()
            emb.append(embeddings)
            torch.cuda.empty_cache()
    # Reshape each element in the emb list to have a consistent shape
    emb = [element.reshape(-1) for element in emb]

    # converts the embeddings into a tensor and reshapes them to the correct size
    emb = np.array(emb)
    emb = np.vstack(emb)

    embeddings = torch.tensor(emb)

    total_samples = int(10 * embeddings.shape[0] / (36))
    embeddings = torch.reshape(embeddings, (total_samples, 36 * 3))
    print("uncertainty shape:", embeddings.shape[0])

    print(len(embeddings))

    return embeddings

def main():
    # unexplained dataset embedding
    datanoexp_path = "./data/dataset_noexp.csv"
    noexp_df = pd.read_csv(datanoexp_path)
    # prepare for uncertainty dataset that can be used in the pre-trained model (the data set without using accurate explanations)
    explanations = read_explanations("explanations.txt")
    df_exp_uncertain = create_explanations_dataset(noexp_df, explanations)

    # default explained data can be passed through the pre-trained model
    # each subset is created and then saved
    print("len df_exp_uncertain", len(df_exp_uncertain))
    print("after standarlise:", (len(df_exp_uncertain) // 360) * 360)
    uncertainset = df_exp_uncertain[0:(len(df_exp_uncertain) // 360) * 360]
    uncertainset.to_csv("./data/uncertainset.csv", index=False)
    uncertainset_dict = load_dataset("csv", data_files="./data/uncertainset.csv")
    directory = './data/exp/uncertainset'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and its contents recursively
            shutil.rmtree(file_path)
    # Save the DatasetDict to disk
    uncertainset_dict.save_to_disk("./data/exp/uncertainset")
    uncertain_raw_dataset = load_from_disk("./data/exp/uncertainset")
    unexpembedding = preprocess_samples_unxep(uncertain_raw_dataset)
    save_filename = (
        "./unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
    )
    save_directory = "./unexp_embeddings/NEW_bertie_embeddings_textattack"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(unexpembedding, save_filename)


if __name__ == "__main__":
    main()
