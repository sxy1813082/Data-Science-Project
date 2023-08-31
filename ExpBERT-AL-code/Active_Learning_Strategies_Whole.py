import math
import os
from random import random
import random

from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import pandas as pd
import emoji
import numpy as np
import torch
import argparse
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import ConcatDataset
import shutil
import torch.utils.data as data_utils

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing
from torch import nn, multiprocessing
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Callable
from math import floor
import seaborn as sns
import matplotlib.pyplot as plt
import openai

random.seed(42)

# iteration steps global t
t = 0

# pass parameter into code
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="the dataset to run: active-dataset, full-data"
)
parser.add_argument(
    "--annotator",
    type=str,
    required=False,
    default="preset",
    help="annotator type: preset, human, openai",
)
parser.add_argument(
    "--strategy",
    type=str,
    required=False,
    default="ds",
    help="active learning strategies type: rs, us, ds",
)
args = parser.parse_args()

def obtain_filepaths(dir_path):
    filepaths = []
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)):
            if file.endswith(".tsv"):
                filepaths.append(os.path.join(dir_path, file))
    return filepaths


# helper function - splits up hashtags into separate words
def camel_case_split(word):
    start_idx = [i for i, e in enumerate(word) if e.isupper()] + [len(word)]
    start_idx = [0] + start_idx
    list_words = [word[x:y] for x, y in zip(start_idx, start_idx[1:])][1:]
    return " ".join(list_words)


# helper function - replaces emojis with textual descriptions of them
def emoji_present(text):
    if (
        emoji.is_emoji(text)
        or (len(text) > 1 and emoji.is_emoji(text[0]))
        or (len(text) > 1 and emoji.is_emoji(text[-1]))
    ):
        emoji_text = emoji.demojize(text)
        emoji_text = emoji_text.replace(":", "")
        emoji_text = emoji_text.replace("_", " ")
        return emoji_text
        # return demoji.replace_with_desc(text, sep="") # had to change this line as demoji not available on conda
    else:
        return text


# cleans the tweet, removing usernames, weblinks and calling helper functions
def placeholders(texts):
    for count, text in enumerate(texts):
        new_text = []
        for t in text.split(" "):
            t = "" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            t = "" if "RT" in t else t
            t = camel_case_split(t) if t.startswith("#") and len(t) > 1 else t
            t = emoji_present(t)
            new_text.append(t)

        texts[count] = " ".join(new_text).strip()

    return texts


# drops unnecessary columns, converts class labels to numbers and cleans tweets
def clean_individual_dataset(filepath):
    df = pd.read_csv(filepath, sep="\t", header=0)
    df.index.name = "Index"

    labels = {
        "injured_or_dead_people": 0,
        "missing_trapped_or_found_people": 1,
        "displaced_people_and_evacuations": 2,
        "infrastructure_and_utilities_damage": 3,
        "donation_needs_or_offers_or_volunteering_services": 4,
        "caution_and_advice": 5,
        "sympathy_and_emotional_support": 6,
        "other_useful_information": 7,
        "not_related_or_irrelevant": 8,
    }
    df = df.drop(columns=["tweet_id"])

    # convert labels to numbers
    df.replace(to_replace={"label": labels}, inplace=True)
    df = df.astype({"tweet_text": "string"})

    # clean the tweets
    df["tweet_text"] = placeholders(df["tweet_text"])

    # remove empty tweets
    df["tweet_text"].replace("", np.nan, inplace=True)
    df.dropna(subset=["tweet_text"], inplace=True)

    return df


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
    if len(explanations) == 0:
        ex_td = textual_descriptions
    else:
        ex_td = textual_descriptions+explanations
    len_df = len(df.index)

    # creates N copies of 'ex_td' where N is the number of tweets
    df = df.iloc[np.repeat(np.arange(len(df)), len(ex_td))].reset_index(drop=True)
    ex_td = ex_td * len_df

    # adds each explanation and textual description to each tweet
    df.insert(1, "exp_and_td", ex_td, allow_duplicates=True)

    return df, len(textual_descriptions)

# helper function - reads the explanations from a text file
def read_explanations(explanation_file):
    f = open(explanation_file, "r")
    lines = f.readlines()
    explanations = [line.strip() for line in lines]
    return explanations

# classify -- ---------------------
torch.multiprocessing.set_sharing_strategy("file_system")
# Setting up the tensorboard for visualising results --------------------
tensorboard_filepath = (
        # "rs_add_no_818"
    # "all_data_818"
    # "rs_add_1_818"
    # "us_add_1_818"
    # "ds_add_1_818"
    args.strategy+"_add_exp_"+args.annotator+"_dataset_"+args.dataset
)
print(tensorboard_filepath)
writer = SummaryWriter(tensorboard_filepath, flush_secs=5)


# Creates dataset given a list of indices --------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, embeddings_IDs, labels):
        self.embeddings_IDs = embeddings_IDs
        self.labels = labels

    def __len__(self):
        return len(self.embeddings_IDs)

    def __getitem__(self, index):
        # Returns subset of embeddings and labels if provided with list
        # of indices
        if type(index) == list:
            embeddings = []
            labels = []
            for val in index:
                embeddings.append(self.embeddings_IDs[val])
                labels.append(self.labels[val])
            return embeddings, labels
        else:
            return self.embeddings_IDs[index], self.labels[index]

# Retrieves the embeddings from the given file path and return training, validation and test --------------------
def get_datasets(originlen,addlen):
    with torch.no_grad():
        orgfile_path = "./embeddings/NEW_bertie_embeddings_textattack/" + "bert-base-uncased-MNLI_subset_1.pt"
        embeddings = torch.load(orgfile_path)

        raw_dataset = load_from_disk("./data/no/")
        labels = np.array(raw_dataset["train"]["labels"])

        # give the index of train and validation dataset
        idx = np.arange(0, len(embeddings), dtype=np.intc)

        embeddings = embeddings[idx]
        labels = labels[idx]

        # load test dataset
        test_raw_dataset = load_from_disk("./testdata/")
        test_labels = test_raw_dataset["train"]["labels"]
        test_file_path = "./test_embeddings/NEW_bertie_embeddings_textattack/" + "bert-base-uncased-MNLI_subset_1.pt"
        test_embedding = torch.load(test_file_path)
        dataset = Dataset(test_embedding, test_labels)
        test_dataset = Dataset(
            dataset[:][0],
            dataset[:][1]
        )
        # load val dataset
        val_raw_dataset = load_from_disk("./valdata/")
        val_labels = val_raw_dataset["train"]["labels"]
        val_file_path = "./val_embeddings/NEW_bertie_embeddings_textattack/" + "bert-base-uncased-MNLI_subset_1.pt"
        val_embedding = torch.load(val_file_path)
        dataset = Dataset(val_embedding, val_labels)
        val_dataset = Dataset(
            dataset[:][0],
            dataset[:][1]
        )

    train_dataset = Dataset(embeddings[idx], labels[idx])
    return train_dataset, val_dataset, test_dataset


# Calculates the performance metrics using the sk-learn library, given predicted and true labels --------------------
def get_metrics(y_preds, y_true):
    accuracy = accuracy_score(y_true, y_preds)
    new_list = list(range(9))
    precision = precision_score(
        y_true, y_preds, labels=new_list, average=None, zero_division=0
    )

    recall = recall_score(
        y_true, y_preds, labels=new_list, average=None, zero_division=0
    )

    f1 = f1_score(y_true, y_preds, labels=new_list, average=None, zero_division=0)

    f1_weighted = f1_score(
        y_true, y_preds, labels=new_list, average="weighted", zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_preds, labels=new_list, average="macro", zero_division=0
    )

    results = {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return results, f1_weighted, f1_macro


# Trainer class, the main part of classifying tweets --------------------
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        # unlabel_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        writer: SummaryWriter,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.unlabel_loader = unlabel_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.writer = writer

    # training the model
    def train(self, epochs: int,
              # active_learning_epochs: list,
              print_frequency: int = 20, start_epoch: int = 0):
        self.model.train()

        progress_bar = tqdm(range(epochs))

        for epoch in range(start_epoch, epochs):
            self.model.train()
            train_logits = []
            train_preds = []
            train_labels = []
            total_training_loss = 0
            for batch, labels in self.train_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                logits = self.model.forward(batch)

                train_logits.append(logits.detach().cpu().numpy())
                train_labels.append(labels.cpu().numpy())

                loss = self.criterion(logits, labels.long())
                total_training_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)

                preds = logits.argmax(-1)
                train_preds.append(preds.cpu().numpy())

            average_training_loss = total_training_loss / len(self.train_loader)

            # calls validate function every print_frequency epochs
            if ((epoch + 1) % print_frequency) == 0:
                preds = logits.argmax(-1)
                accuracy = accuracy_score(
                    labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
                )
                train_accuracy = accuracy * 100

                (
                    val_epoch_metrics,
                    average_val_loss,
                    val_accuracy,
                    f1_weighted,
                    f1_macro,
                ) = self.validate()

                # adds values to tensorboard for each epoch not active learning iteration
                self.writer.add_scalars(
                    "accuracy", {"train": train_accuracy, "val": val_accuracy}, epoch
                )
                self.writer.add_scalars(
                    "loss",
                    {"train": average_training_loss, "val": average_val_loss},
                    epoch,
                )
                self.writer.add_scalars(
                    "f1_score", {"weighted": f1_weighted, "macro": f1_macro}, epoch
                )

                self.model.train()  # Need to put model back into train mode after evaluating

            # calls test function in the final epoch
            if epoch == (epochs - 1):
                global t
                print("epoch number", epoch)
                preds = logits.argmax(-1)
                accuracy = accuracy_score(
                    labels.detach().cpu().numpy(), preds.detach().cpu().numpy()
                )
                self.writer.add_scalars("train dataset performance",
                                        {"acc": accuracy*100}, t)

                (
                    test_data_metrics,
                    average_test_data_loss,
                    test_data_accuracy,
                    test_f1_weighted,
                    test_f1_macro,
                ) = self.test()  # Run test set
                (
                    metric_results, average_loss, val_accuracy, f1_weighted, f1_macro,
                ) = self.validate()

                print("global t is :",t)
                # this is for record the real performance in active learning fot each iteration
                self.writer.add_scalars("test performance", {"test_acc":test_data_metrics['accuracy'],"f1_marco":test_f1_macro}, t)
                self.writer.add_scalars("validation dataset performance",{"val_acc":val_accuracy,"f1_marco":f1_macro*100}, t)
                print("test epoch results", test_data_metrics, flush=True)
                print("val epoch results", metric_results, flush=True)

            self.step += 1

    def validate(self):
        results = {"preds": [], "labels": []}

        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels.long())
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        average_loss = total_loss / len(self.val_loader)
        val_accuracy = accuracy_score(results["labels"], results["preds"]) * 100

        metric_results, f1_weighted, f1_macro = get_metrics(
            results["preds"], results["labels"]
        )

        return metric_results, average_loss, val_accuracy, f1_weighted, f1_macro

    # called in the final epoch to evaluate the trained model
    def test(self):
        results = {"preds": [], "labels": []}

        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for batch, labels in self.test_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels.long())
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        average_loss = total_loss / len(self.test_loader)
        test_accuracy = accuracy_score(results["labels"], results["preds"]) * 100

        metric_results, f1_weighted, f1_macro = get_metrics(
            results["preds"], results["labels"]
        )

        # creates a confusion matrix given the predictions and true labels. This is only done for the test data.
        conf_matrix = confusion_matrix(
            results["labels"], results["preds"], normalize="true"
        )
        df_cm = pd.DataFrame(conf_matrix)

        plt.figure(figsize=(10, 8))
        conf_heatmap = sns.heatmap(df_cm, annot=True)
        conf_heatmap.set(
            xlabel="Predicted Label", ylabel="True Label", title="Confusion Matrix"
        )
        fig = conf_heatmap.get_figure()
        fig.savefig("conf_mat.png")
        self.writer.add_figure("Confusion Matrix", conf_heatmap.get_figure())

        return metric_results, average_loss, test_accuracy, f1_weighted, f1_macro


# Initialising the neural network --------------------
class MLP_1h(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layer_size: int,
        output_size: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x

# this is the preset annotator
def generate_explanations(sampled_data):

    strings = []
    global t
    with open("annotator.txt", 'r') as file:
        lines = file.readlines()
    start_line = t * 1
    end_line = (t + 1) * 1
    strings = lines[start_line:end_line]
    print(strings)
    string_array = [string.strip() for string in strings]
    return string_array

# this is for human annotator
def generate_explanations_human(sampled_data):
    strings = []
    global t
    print(t)
    target_texts = [data["text"] for data in sampled_data if data["labels"] == t]

    if len(target_texts) > 10:
        target_texts = target_texts[:10]

    print(f"Label: {t}")
    print("Sampled Texts:")

    for text in target_texts:
        print(text)

    user_input = input("please input a key explanation for the texts ")
    strings.append(str(user_input))

    # Convert list to string for printing
    explanations = ', '.join(strings)
    print("explanation is: "+explanations)
    strings.append(str(explanations))

    return strings

# this is the OpenAI annotator
def generate_explanations_AI(sampled_data):
    strings = []
    global t
    textual_descriptions = [
        "hurt or dead",
        "missing or trapped or found",
        "displaced and evacuations",
        "infrastructure or utilities damage",
        "donation or offers or volunteering",
        "caution and advice",
        "sympathy and emotional support",
        "other useful information",
        "not related or irrelevant",
    ]
    # Getting texts with the target label
    target_texts = [data["text"] for data in sampled_data if data["labels"] == t]

    # Limit to 10 texts if there are more
    if len(target_texts) > 5:
        target_texts = target_texts[:5]

    print(f"Label: {t}")

    # Generate one long text from the samples
    x = "generate a string that only contains the words in the text that are only semantically equal to "+textual_descriptions[t]+": "+"\n\n"
    long_text = x + " ".join(target_texts)+"."+"\n\n"

    print(long_text)

    # Ask OpenAI model for an explanation
    openai.api_key = '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'
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the Ada engine
        prompt=long_text,
        temperature=1,
        max_tokens=25,
        top_p=1,
        n=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    explanation = str(response.choices[0].text.strip())
    print("AI Explanation:", explanation)
    strings.append(explanation)
    return strings

# add or delete explained data
def addOrDelete(sampled_indices,raw_dataset_noexp):
    sampled_data = [raw_dataset_noexp['train'][i] for i in sampled_indices]
    no_exp_data = []
    exp_list = []
    new_data = []

    # give explanation to the sampled data
    if(args.annotator == "preset"):
        exp_list = generate_explanations(sampled_data)
    if(args.annotator == "openai"):
        exp_list = generate_explanations_AI(sampled_data)
    if(args.annotator == "human"):
        exp_list = generate_explanations_human(sampled_data)

    # exp list store in txt file
    file_path = "explanations.txt"

    # Open the file in write mode
    with open(file_path, 'a') as file:
        # Iterate over each item in exp_list and write it to the file
        for item in exp_list:
            file.write(str(item) + '\n')

    # Close the file
    file.close()

    # Standardise the sampled data and join all the explanations to the text
    for data in sampled_data:
        tweet = data["text"]
        label = data["labels"]
        if tweet == "" or label == "":
            continue
        tweet = placeholders([tweet])
        tweet = tweet[0]
        no_exp_data.append({"text": tweet, "labels": label})
        explanation = exp_list
        with open("explanations.txt", "r") as file:
            # explanations.append(exp)
            extracted_explanations = file.readlines()[:]
            for extracted_explanation in extracted_explanations:
                new_data.append({"text": tweet, "labels": label, "exp_and_td": extracted_explanation.strip()})
        with open("DefaultuseExp.txt", "r") as file:
            extracted_explanations = file.readlines()[:]
            for extracted_explanation in extracted_explanations:
                new_data.append({"text": tweet, "labels": label, "exp_and_td": extracted_explanation.strip()})
    new_data = {"text": [data["text"] for data in new_data],
                "labels": [data["labels"] for data in new_data],
                "exp_and_td": [data["exp_and_td"] for data in new_data]}
    new_data = pd.DataFrame(new_data)
    new_data.to_csv("./data/dataset_exp_new_set.csv", index=False)
    # Load the CSV file into a DatasetDict
    new_dict = load_dataset("csv", data_files="./data/dataset_exp_new_set.csv")
    directory = './data/new_exp/new'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and its contents recursively
            shutil.rmtree(file_path)
    # Save the DatasetDict to disk
    new_dict.save_to_disk("./data/new_exp/new")

    #==============================annotated dataset update=====================
    df_noexp = pd.read_csv("./data/dataset_noexp_no.csv")
    explanations = read_explanations("explanations.txt")
    # reconnect the new explanation set
    df_exp, num_des = create_explanations_dataset(df_noexp, explanations)
    #new annotated data can be passed through the pre-trained model
    num = len(explanations) + num_des
    nums = num * 30*3
    subset_1 = df_exp[:]
    subset_1.to_csv("./data/dataset_exp_subset_1.csv", index=False)
    existing_data_path = "./data/dataset_exp_subset_1.csv"
    existing_data_df = pd.read_csv(existing_data_path)
    new_data_df = pd.DataFrame(new_data)
    merged_data_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
    merged_data_sub = merged_data_df[0:(len(merged_data_df) // nums) * nums]
    merged_data_sub.to_csv(existing_data_path, index=False)
    # Load the CSV file into a DatasetDict
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_1.csv")
    directory = './data/exp/subset_1'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and its contents recursively
            shutil.rmtree(file_path)
    # Save the DatasetDict to disk
    subset_1_dict.save_to_disk("./data/exp/subset_1")

    # Extend the dataset that have the default labels by adding the explained data(without explanation) into the origin dataset
    # This is for the get labels in the training functions for NN classifier
    new_data_no = {"text": [data["text"] for data in no_exp_data],
                   "labels": [data["labels"] for data in no_exp_data]}
    existing_data_path = "./data/dataset_noexp_no.csv"
    existing_data_df = pd.read_csv(existing_data_path)
    new_data_df = pd.DataFrame(new_data_no)
    merged_data_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
    merged_data_df.to_csv(existing_data_path, index=False)
    # Load the CSV file into a DatasetDict
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_noexp_no.csv")
    directory = './data/no/'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and its contents recursively
            shutil.rmtree(file_path)
    # Save the DatasetDict to disk
    subset_1_dict.save_to_disk("./data/no/")
    # ===========================================================================

    # Delete the explained data from unannotated dataset========================
    # This is for the unexplained dataset
    orgfile_path = "./unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
    new_path = "./new_unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
    save_directory = "./new_unexp_embeddings/NEW_bertie_embeddings_textattack"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if t==0:
        unexp_embeddings = torch.load(orgfile_path)
        # Find the indices that are not in sampled_indices
        indices_to_keep = [i for i in range(len(unexp_embeddings)) if i not in sampled_indices]
        # Select the embeddings that correspond to the indices to keep
        filtered_embeddings = unexp_embeddings[indices_to_keep]
        torch.save(filtered_embeddings, new_path)
    else:
        unexp_embeddings = torch.load(new_path)
        # Find the indices that are not in sampled_indices
        indices_to_keep = [i for i in range(len(unexp_embeddings)) if i not in sampled_indices]
        # Select the embeddings that correspond to the indices to keep
        filtered_embeddings = unexp_embeddings[indices_to_keep]
        torch.save(filtered_embeddings, new_path)
    # Find the indices that are not in sampled_indices
    sampled_data_no = [data for i, data in enumerate(raw_dataset_noexp['train']) if i not in sampled_indices]
    data_no_exp_new = []
    for data in sampled_data_no:
        tweet = data["text"]
        label = data["labels"]
        data_no_exp_new.append({"text": tweet, "labels": label})
    new_data_no_exp = {"text": [data["text"] for data in data_no_exp_new],
                       "labels": [data["labels"] for data in data_no_exp_new]}
    existing_data_path = "./data/dataset_noexp.csv"
    new_data_df = pd.DataFrame(new_data_no_exp)
    new_data_df.to_csv(existing_data_path, index=False)
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_noexp.csv")
    directory = './data/org/'
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Remove the subdirectory and its contents recursively
            shutil.rmtree(file_path)
    # Save the DatasetDict to disk
    subset_1_dict.save_to_disk(directory)
    # =============================================================

# Process a single batch and return the embeddings
def process_batch(batch, model, tokenizer, tokenize_exp_function):
    with torch.no_grad():
        tokenized_train = tokenize_exp_function(batch, tokenizer)
        model_outputs = model(**tokenized_train)
        embeddings = model_outputs["logits"]
        embeddings = embeddings.cpu().detach().numpy()
        return embeddings

def tokenize_exp_function(examples,tokenizer):
            return tokenizer(
                examples["text"],
                examples["exp_and_td"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

# takes in only the tweet and creates an embedding
def tokenize_noexp_function(examples,tokenizer):
    return tokenizer(
        examples["text"], truncation=True, padding=True, return_tensors="pt"
    )



def least_confidence(prob_dist, sorted=False):
    """
    Keyword arguments:
        prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
        sorted -- if the probability distribution is pre-sorted from largest to smallest
    """
    if sorted:
        simple_least_conf = prob_dist[0]  # most confident prediction
    else:
        simple_least_conf = torch.max(prob_dist)  # most confident prediction
    num_labels = prob_dist.numel()  # number of labels
    normalized_least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
    return normalized_least_conf.item()

def uncertainty_sampling(model, k,num):
    # Preprocess the raw dataset
    if t==0:
        orgfile_path = "./unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
        embeddings = torch.load(orgfile_path)
    else:
        orgfile_path = "./new_unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
        embeddings = torch.load(orgfile_path)

    target_shape = (len(embeddings), num*3)
    pad_amount = max(target_shape[1] - embeddings.shape[1], 0)
    embeddings = F.pad(embeddings, (0, pad_amount))
    # Set the model to evaluation model
    model.eval()
    with torch.no_grad():
        # Make predictions using the model
        logits = model(embeddings)  # Assuming the model returns logits
        # Calculate the prediction probabilities
        probs = torch.softmax(logits, dim=-1)
        # Calculate the uncertainty scores using least confidence
        uncertainty_scores = [least_confidence(prob_dist) for prob_dist in probs]
        # Rank the samples based on the uncertainty scores
        sorted_indices = np.argsort(uncertainty_scores)
        # Select the top k least confident samples
        least_confident_indices = sorted_indices[-k:]
    # Return the indices of the selected samples
    return least_confident_indices.tolist()

def semantic_diversity_sampling(model, k, num):
    # Preprocess the raw dataset
    if t == 0:
        orgfile_path = "./unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
        embeddings = torch.load(orgfile_path)
    else:
        orgfile_path = "./new_unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
        embeddings = torch.load(orgfile_path)
    target_shape = (len(embeddings), num * 3)
    pad_amount = max(target_shape[1] - embeddings.shape[1], 0)
    embeddings = F.pad(embeddings, (0, pad_amount))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Make predictions using the model
        logits = model(embeddings)  # Assuming the model returns logits
        # Calculate the prediction probabilities
        probs = torch.softmax(logits, dim=-1)
        # Initialize cluster centers by selecting α vectors from Yi 4% each iteration∗
        # please change alpha according to your data size
        alpha = 0.2
        embeddings = embeddings.numpy()
        cluster_centers = embeddings[np.random.choice(len(embeddings), int(alpha * len(embeddings)), replace=False)]

        # Calculate distances from each embedding to each cluster center
        distances = np.zeros((len(embeddings), len(cluster_centers)))
        for i, embedding in enumerate(embeddings):
            for j, center in enumerate(cluster_centers):
                distances[i, j] = euclidean(embedding, center)

        # Initialize a list to store the selected indices
        selected_indices = []

        # Select the first center index with the maximum distance
        initial_center_index = np.argmax(np.max(distances, axis=1))
        selected_indices.append(initial_center_index)

        # Update the cluster center with the initial selected embedding
        cluster_centers[0] = embeddings[initial_center_index]

        # Loop through the remaining cluster centers
        for c in range(1, len(cluster_centers)):
            # Calculate the distances from the embeddings to the current cluster center
            distances = np.array(
                [np.min(np.linalg.norm(cluster_centers[:c] - embedding, axis=1)) for embedding in embeddings])
            # Find the index of the embedding with the maximum distance to the existing centers
            new_center_index = np.argmax(distances)
            # Add the new center to the cluster centers list
            cluster_centers[c] = embeddings[new_center_index]
            # Add the index of the selected embedding to the list of selected_indices
            selected_indices.append(new_center_index)
        selected_indices = [int(i) for i in selected_indices]
        # Return the indices of the selected samples
        return selected_indices

# pre-train the unannotated dataset
def preprocess_samples_unxep(raw_dataset_noexp):
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    if not os.path.exists("unexp_embeddings"):
        os.makedirs("unexp_embeddings")
    emb = []
    train_dataloader = DataLoader(raw_dataset_noexp["train"], batch_size=10)
    for batch in train_dataloader:
        with torch.no_grad():
            tokenized_train = tokenize_noexp_function(batch,tokenizer)
            model_outputs = model(**tokenized_train)
            embeddings = model_outputs["logits"]
            embeddings = embeddings.cpu().detach().numpy()
            emb.append(embeddings)
            torch.cuda.empty_cache()
    # Reshape each element in the emb list to have a consistent shape that can be input into the NN classifier
    emb = [element for element in emb]
    emb = np.vstack(emb)
    explanations = read_explanations("explanations.txt")
    num = len(explanations)+9
    embeddings = torch.tensor(emb)

    target_shape = (len(embeddings), 27)
    pad_amount = max(target_shape[1] - embeddings.shape[1], 0)
    padded_embeddings = F.pad(embeddings, (0, pad_amount))

    return padded_embeddings

def preprocess_samples(raw_dataset_noexp,num):
    print("preprocess_samples annotated set begin ...")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
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
    total_samples = int(embeddings.shape[0] * embeddings.shape[1] / (num * 3))
    embeddings = torch.reshape(embeddings, (total_samples, num * 3))
    return embeddings

def main():
    file_path = "explanations.txt"
    with open(file_path, 'w') as file:
        file.truncate(0)
    # prepare dataset and train and validation embedding
    dataframes = []
    filepaths = obtain_filepaths("./data/")

    # cleans the data from each disaster individually
    for file in filepaths:
        df = clean_individual_dataset(file)
        dataframes.append(df)

    # concatenates the tweets from each disaster to form one dataset
    df_concat = pd.concat(dataframes)

    # renames the columns
    df_concat.rename(columns={"tweet_text": "text"}, inplace=True)
    df_concat.rename(columns={"label": "labels"}, inplace=True)

    # duplicate tweets are dropped
    df_noexp_all = df_concat.drop_duplicates(subset=["text"], inplace=False)

    # Use StratifiedShuffleSplit for stratified sampling
    # this is the active-learning dataset
    if(args.dataset == "active-dataset"):
        X = df_noexp_all.drop('labels', axis=1)
        y = df_noexp_all['labels']
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
        train_index, other_index = next(split.split(X, y))
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.875, random_state=42)
        val_index, test_index = next(split.split(X.iloc[other_index], y.iloc[other_index]))
        val_index = other_index[val_index]
        test_index = other_index[test_index]
        print("len test:",len(val_index))
        print("len train and val",len(train_index))
        print("len unannotated set ",len(test_index))
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        trn_index, validation_index = next(split.split(X.iloc[train_index], y.iloc[train_index]))
        print("len train",len(trn_index))
        print("len val",len(validation_index))

    # # use full-data dataset
    if(args.dataset == "full-data"):
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_index, other_index = next(split.split(X, y))
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=42)
        val_index, test_index = next(split.split(X.iloc[other_index], y.iloc[other_index]))
        val_index = other_index[val_index]
        test_index = other_index[test_index]
        print("len test:", len(val_index))
        print("len train*val", len(train_index))
        print("len unxep", len(test_index))
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.07, random_state=42)
        trn_index, validation_index = next(split.split(X.iloc[train_index], y.iloc[train_index]))
        print("len train set (annotated dataset)", len(trn_index))
        print("len val", len(validation_index))


    # Split the data into four parts based on the indices
    # train (annotated dataset)
    df_noexp = df_noexp_all.iloc[trn_index]
    # unannotated dataset
    df_noexp_two = df_noexp_all.iloc[test_index]
    # test dataset
    test_noexp_all = df_noexp_all.iloc[val_index]
    # validation dataset
    val_noexp_all = df_noexp_all.iloc[validation_index]


    # test and validation dataset
    test_noexp_all.to_csv("./test_data/dataset_noexp.csv", index=False)
    val_noexp_all.to_csv("./val_data/dataset_noexp.csv",index=False)

    # saves the train(unannotated) dataset to a dataset directory and pre-train------
    df_noexp_two.to_csv("./data/dataset_noexp.csv", index=False)
    data_noexp = load_dataset("csv", data_files="./data/dataset_noexp.csv")
    data_noexp.save_to_disk("./data/org/")
    uncertain_raw_dataset = load_from_disk("./data/org/")

    unexpembedding = preprocess_samples_unxep(uncertain_raw_dataset)
    save_filename = (
        "./unexp_embeddings/NEW_bertie_embeddings_textattack/unexp.pt"
    )
    save_directory = "./unexp_embeddings/NEW_bertie_embeddings_textattack"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(unexpembedding, save_filename)
    # -------------------------------------------------------------------------------------

    # annotated dataset connect to explanations and save-------------------------------------------------
    df_noexp.to_csv("./data/dataset_noexp_no.csv", index=False)
    data_noexp_one = load_dataset("csv", data_files="./data/dataset_noexp_no.csv")
    data_noexp_one.save_to_disk("./data/no/")
    originlen = len(data_noexp_one["train"]["labels"])

    # reads in explanations and concatenates to the tweets to form an expanded dataset (to initial the pretrained model)
    explanations = read_explanations("explanations.txt")
    df_exp, num_des= create_explanations_dataset(df_noexp, explanations)

    num = len(explanations) + num_des
    nums = num * 30*3
    subset_1 = df_exp[0:(len(df_exp) // nums) * nums]
    subset_1.to_csv("./data/dataset_exp_subset_1.csv", index=False)
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_1.csv")
    subset_1_dict.save_to_disk("./data/exp/subset_1")
    temp_path_noexp = "./data/org/"
    raw_dataset_noexp = load_from_disk(temp_path_noexp)
    datanoexp_path = "./data/dataset_noexp.csv"
    noexp_df = pd.read_csv(datanoexp_path)

    # human in the loop begin and stop when t equals to 9
    global t
    addlen = 0
    while noexp_df.shape[0] > 0 and t<10:
        temp_path_noexp = "./data/org/"
        raw_dataset_noexp = load_from_disk(temp_path_noexp)
        datanoexp_path = "./data/dataset_noexp.csv"
        noexp_df = pd.read_csv(datanoexp_path)
        explanations = read_explanations("explanations.txt")
        num = len(explanations) + 9
        nums = num * 30 * 3

        #pre-train train dataset in each iteration ------------------------------------------------------
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")

        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # optimize the model to cpu running when the work is apply to GPU
        model = model.to(device)
        torch.cuda.empty_cache()

        # splits the dataset into batches of size 10 and passes them through the tokenizer and pre-trained model.
        print("embedding begin")
        emb = []
        temp_path = "./data/exp/subset_1"
        raw_dataset = load_from_disk(temp_path)
        train_dataloader = DataLoader(raw_dataset["train"], batch_size=10)
        model.eval()

        # use pool to embedding the instances
        pool = multiprocessing.Pool()
        for batch in train_dataloader:
            result = pool.apply_async(process_batch, (batch, model, tokenizer, tokenize_exp_function))
            emb.append(result.get())
        pool.close()
        pool.join()

        # Reshape each element in the emb list to have a consistent shape
        emb = [element.reshape(-1) for element in emb]
        # Initialize a list to store the padded arrays
        padded_emb = []
        # Desired size after padding
        desired_size = 30
        # Loop through each element in emb
        for element in emb:
            current_size = element.shape[0]
            if current_size < desired_size:
                # Calculate the amount of padding needed
                pad_amount = desired_size - current_size
                # Pad the array with zeros at the end (you can use any other value for padding as needed)
                padded_array = np.pad(element, (0, pad_amount), mode='constant')
                padded_emb.append(padded_array)
            else:
                # If the array is already of the desired size, no padding is needed
                padded_emb.append(element)
        # converts the embeddings into a tensor and reshapes them to the correct size
        emb = np.array(padded_emb)
        emb = np.vstack(emb)
        embeddings = torch.tensor(emb)
        total_samples = int(embeddings.shape[0] * embeddings.shape[1] / (num * 3))
        embeddings = torch.reshape(embeddings, (total_samples, num * 3))

        # and then saves the embedding
        save_filename = ("./embeddings/NEW_bertie_embeddings_textattack/bert-base-uncased-MNLI_subset_1.pt")
        save_directory = "./embeddings/NEW_bertie_embeddings_textattack"
        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(embeddings, save_filename)

        # pre-train  test dataset----------------------------------------------------
        explanations = read_explanations("explanations.txt")
        data_noexp = load_dataset("csv", data_files="./test_data/dataset_noexp.csv")
        directory = "./testdata/"
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Remove the subdirectory and its contents recursively
                shutil.rmtree(file_path)
        # Save the DatasetDict to disk
        data_noexp.save_to_disk("./testdata/")

        test_exp, num_texdes = create_explanations_dataset(test_noexp_all, explanations)
        subset_test = test_exp[0:(len(test_exp) // nums) * nums]
        subset_test.to_csv("./data/dataset_exp_subset_test.csv", index=False)
        subset_test_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_test.csv")
        directory = "./data/exp/subset_test"
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Remove the subdirectory and its contents recursively
                shutil.rmtree(file_path)
        # Save the DatasetDict to disk
        subset_test_dict.save_to_disk("./data/exp/subset_test")
        test_raw_dataset = load_from_disk("./data/exp/subset_test")
        test_embedding = preprocess_samples(test_raw_dataset,num)
        save_filename = ("./test_embeddings/NEW_bertie_embeddings_textattack/bert-base-uncased-MNLI_subset_1.pt")
        save_directory = "./test_embeddings/NEW_bertie_embeddings_textattack"
        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(test_embedding, save_filename)

        # -----------------------------pretrain validation dataset--------------------
        explanations = read_explanations("explanations.txt")
        data_noexp = load_dataset("csv", data_files="./val_data/dataset_noexp.csv")
        directory = "./valdata/"
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Remove the subdirectory and its contents recursively
                shutil.rmtree(file_path)
        # Save the DatasetDict to disk
        data_noexp.save_to_disk("./valdata/")

        val_exp, num_texdes = create_explanations_dataset(val_noexp_all, explanations)
        subset_val = val_exp[0:(len(val_exp) // nums) * nums]
        subset_val.to_csv("./data/dataset_exp_subset_val.csv", index=False)
        subset_val_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_val.csv")
        directory = "./data/exp/subset_val"
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                # Remove the file
                os.remove(file_path)
            elif os.path.isdir(file_path):
                # Remove the subdirectory and its contents recursively
                shutil.rmtree(file_path)
        # Save the DatasetDict to disk
        subset_val_dict.save_to_disk("./data/exp/subset_val")
        val_raw_dataset = load_from_disk("./data/exp/subset_val")
        val_embedding = preprocess_samples(val_raw_dataset,num)
        save_filename = ("./val_embeddings/NEW_bertie_embeddings_textattack/bert-base-uncased-MNLI_subset_1.pt")
        save_directory = "./val_embeddings/NEW_bertie_embeddings_textattack"
        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(val_embedding, save_filename)

        #classifiy the tweets---------------------------------------------------
        print("classify begin")
        data_noexp_one = load_from_disk("./data/no/")
        originlen = len(data_noexp_one["train"]["labels"])
        train_dataset, val_dataset, test_dataset = get_datasets(originlen,addlen)

        # DataLoader splits the datasets into batches
        train_loader = DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=8,
            num_workers=2,
            pin_memory=True,
        )

        # optimises the training if running on a GPU
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        feature_count = train_dataset[0][0].shape[0]
        class_count = 9

        torch.manual_seed("37")

        # initialises the model and hyperparameters taking into account the passed in arguments
        model_NN = MLP_1h(feature_count, int("100"), class_count)

        model_NN = model_NN.to(device)
        optimizer = AdamW(
            model_NN.parameters(), lr=float("5e-5"), weight_decay=float("1e-2")
        )

        criterion = nn.CrossEntropyLoss()

        # initalises the Trainer class
        trainer = Trainer(
            model_NN,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            optimizer,
            device,
            writer,
        )

        # calls train to start the training, validating and testing process
        trainer.train(
            int("30"),
            print_frequency=1,
        )

        writer.close()

        # begin acquisition function
        if(args.strategy == "us"):
            if noexp_df.shape[0] > 500:
                sampled_indices = uncertainty_sampling(model_NN, 500, num)
            else:
                sampled_indices = uncertainty_sampling(model_NN, noexp_df.shape[0], num)
        if (args.strategy == "rs"):
            if noexp_df.shape[0] >= 500:
                sampled_indices = random.sample(range(noexp_df.shape[0]), 500)
            else:
                sampled_indices = random.sample(range(noexp_df.shape[0]), noexp_df.shape[0])
        if (args.strategy == "ds"):
            if noexp_df.shape[0] >= 500:
                sampled_indices = semantic_diversity_sampling(model_NN, 500, num)
            else:
                sampled_indices = semantic_diversity_sampling(model_NN, noexp_df.shape[0], num)

        # add data and delete data  from annotated dataset and unannotated dataset
        addOrDelete(sampled_indices,raw_dataset_noexp)
        datanoexp_path = "./data/dataset_noexp.csv"
        noexp_df = pd.read_csv(datanoexp_path)

        # go to next iteration (repeat the steps)
        t = t+1
        print("iteration step t is :",t)
        # add 4% data each iteration
        addlen = addlen + 500
        print("addlen is: ",addlen)


if __name__ == "__main__":
    main()