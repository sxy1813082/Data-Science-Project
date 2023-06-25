import os
from random import random
import random

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
    ex_td = explanations + textual_descriptions
    len_df = len(df.index)

    # creates N copies of 'ex_td' where N is the number of tweets
    df = df.iloc[np.repeat(np.arange(len(df)), len(ex_td))].reset_index(drop=True)
    ex_td = ex_td * len_df

    # adds each explanation and textual description to each tweet
    df.insert(1, "exp_and_td", ex_td, allow_duplicates=True)

    return df


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
        "bertie_40_5e-5_wd_1e-2_run_1_seed_37_other_other"
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

class MyCustomDataset:
    def __init__(self, text, labels, exp_and_td):
        self.text = text
        self.labels = labels
        self.exp_and_td = exp_and_td

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.labels[index], self.exp_and_td[index]

class MyCustomDatasetNo:
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.labels[index]
    # def add_labeled_samples(self, indices):
    #     self.labeled_indices.extend(indices)
    #
    # def remove_samples(self, indices):
    #     self.labeled_indices = [i for i in self.labeled_indices if i not in indices]
    #
    # def get_labeled_dataset(self):
    #     return Subset(self, self.labeled_indices)


# Retrieves the embeddings from the given file path and splits dataset into training, validation and test --------------------
def get_datasets(raw_dataset_noexp_no=None):
    with torch.no_grad():
        # loading in the embeddings
        file_path = "./embeddings/NEW_bertie_embeddings_textattack/" + "bert-base-uncased-MNLI_subset_1.pt"

        embeddings = torch.load(file_path)
        raw_dataset = load_from_disk("./data/no/")
        labels = np.array(raw_dataset["train"]["labels"])

        # shuffling embeddings and labels
        idx = np.arange(0, len(embeddings), dtype=np.intc)
        np.random.seed(int("37"))
        np.random.shuffle(idx)
        embeddings = embeddings[idx]
        labels = labels[idx]
        dataset = Dataset(embeddings, labels)

        # splitting up the dataset into training, validation and test
        if (0.7 * len(dataset)) % 1 == 0:
            train_size = int(0.7 * len(dataset))
            test_and_val_size = len(dataset) - train_size
            val_size = 0.5 * test_and_val_size
            train_and_val_size = int(floor(train_size + val_size))

            train_dataset = Dataset(
                dataset[:train_size][0],
                dataset[:train_size][1],
            )

            val_dataset = Dataset(
                dataset[train_and_val_size:][0],
                dataset[train_and_val_size:][1],
            )

            test_dataset = Dataset(
                dataset[train_size:train_and_val_size][0],
                dataset[train_size:train_and_val_size][1],
            )

        else:  # used if there is an unequal split e.g. 25.2 and 10.79
            train_size = int(floor(0.7 * len(dataset)))
            test_and_val_size = len(dataset) - train_size
            val_size = 0.5 * test_and_val_size
            train_and_val_size = int(floor(train_size + val_size))

            train_dataset = Dataset(
                dataset[:train_size][0],
                dataset[:train_size][1],
            )

            val_dataset = Dataset(
                dataset[train_and_val_size:][0],
                dataset[train_and_val_size:][1],
            )

            test_dataset = Dataset(
                dataset[train_size:train_and_val_size][0],
                dataset[train_size:train_and_val_size][1],
            )

        print(len(train_dataset))
        print(len(test_dataset))
        print(len(val_dataset))
        # Create a subset of unlabelled data for active learning
        # train_size_half = int(len(train_dataset) / 2)
        # unlabelled_dataset = Dataset(
        #     train_dataset[train_size_half:][0],
        #     train_dataset[train_size_half:][1],
        # )
        # train_dataset = Dataset(
        #     train_dataset[:train_size_half][0],
        #     train_dataset[:train_size_half][1],
        # )
    return train_dataset, val_dataset, test_dataset, idx


# Calculates the performance metrics using the sk-learn library, given predicted and true labels --------------------
def get_metrics(y_preds, y_true):
    accuracy = accuracy_score(y_true, y_preds)

    precision = precision_score(
        y_true, y_preds, labels=list(range(9)), average=None, zero_division=0
    )

    recall = recall_score(
        y_true, y_preds, labels=list(range(9)), average=None, zero_division=0
    )

    f1 = f1_score(y_true, y_preds, labels=list(range(9)), average=None, zero_division=0)

    f1_weighted = f1_score(
        y_true, y_preds, labels=list(range(9)), average="weighted", zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_preds, labels=list(range(9)), average="macro", zero_division=0
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

            # Active learning step
            # Active learning step
            # if epoch + 1 in active_learning_epochs:
            #     self.active_learning_step(len(self.unlabel_loader))

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

                # adds values to tensorboard
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
                print("epoch number", epoch)
                (
                    test_data_metrics,
                    average_test_data_loss,
                    test_data_accuracy,
                    test_f1_weighted,
                    test_f1_macro,
                ) = self.test()  # Run test set
                print("test epoch results", test_data_metrics, flush=True)

            self.step += 1

            # arranges the predictions to have the correct shape
            train_preds = np.concatenate(train_preds).ravel()
            train_labels = np.concatenate(train_labels).ravel()

    # def active_learning_step(self, num_samples_to_label):
    #     self.model.eval()
    #     uncertainties = []
    #
    #     with torch.no_grad():
    #         for batch, _ in self.unlabel_loader:
    #             batch = batch.to(self.device)
    #             logits = self.model(batch)
    #             probs = F.softmax(logits, dim=-1)
    #             entropy = -(probs * torch.log(probs)).sum(dim=-1)
    #             uncertainties.extend(entropy.cpu().numpy())
    #
    #     top_indices = np.argsort(uncertainties)[-num_samples_to_label:]
    #
    #     # Move selected samples from the unlabelled dataset to the labelled dataset
    #     self.train_loader.dataset.add_labeled_samples(top_indices)
    #     self.unlabel_loader.dataset.remove_samples(top_indices)

    # called every print_frequency epochs to evaluate the model during training
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


# Manipulates weights to be passed into WCEL --------------------
def get_weights():
    # Use distribution of labels for weights
    weights = torch.tensor(
        [13.0864, 2.1791, 3.1197, 7.9103, 14.1146, 5.8246, 11.0066, 29.8417, 12.917],
        dtype=torch.float32,
    )
    # class size is inversely proportional to weight of class
    # Convert percentages into correct fractional form e.g. 10% => 0.1
    weights = weights / weights.sum()
    # Make weights inversely proportional to class size
    weights = 1.0 / weights
    # Scale weights so they sum to 1
    weights = weights / weights.sum()
    return weights


def generate_explanations(param):
    # prompt = param  # Modify the prompt as needed
    # openai.api_key = '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'
    # response = openai.Completion.create(
    #     engine="text-davinci-003",  # Choose the appropriate OpenAI engine
    #     prompt=prompt,
    #     max_tokens=15,
    #     n=1,
    #     stop=None,
    #     temperature=0.8,
    #     top_p=1.0,
    #     frequency_penalty=0.0,
    #     presence_penalty=0.0
    # )
    # explanation = response.choices[0].text.strip()
    explanation = "irrelevant and not relative"
    return explanation

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

def main():
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

    # split noexp into two part one part will be explained properly and one part use default explanations
    df_noexp, df_noexp_two = train_test_split(df_noexp_all, test_size=0.4, random_state=42)

    # saves the unexplained dataset to a dataset directory
    df_noexp_two.to_csv("./data/dataset_noexp.csv", index=False)
    data_noexp = load_dataset("csv", data_files="./data/dataset_noexp.csv")
    data_noexp.save_to_disk("./data/org/")

    # to initial the classifier
    df_noexp.to_csv("./data/dataset_noexp_no.csv", index=False)
    data_noexp_one = load_dataset("csv", data_files="./data/dataset_noexp_no.csv")
    data_noexp_one.save_to_disk("./data/no/")

    # reads in explanations and concatenates to the tweets to form an expanded dataset (to initial the pretrained model)
    explanations = read_explanations("explanations.txt")
    df_exp = create_explanations_dataset(df_noexp, explanations)

    #default explained data can be passed through the pre-trained model
    # each subset is created and then saved
    subset_1 = df_exp[0:72000]
    subset_1.to_csv("./data/dataset_exp_subset_1.csv", index=False)
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_1.csv")
    subset_1_dict.save_to_disk("./data/exp/subset_1")


    temp_path_noexp = "./data/org/"
    raw_dataset_noexp = load_from_disk(temp_path_noexp)
    datanoexp_path = "./data/dataset_noexp.csv"
    noexp_df = pd.read_csv(datanoexp_path)

    # human in the loop
    print(len(raw_dataset_noexp["train"]))
    print(noexp_df.shape[0])
    while noexp_df.shape[0] > 661:
        temp_path_noexp = "./data/org/"
        raw_dataset_noexp = load_from_disk(temp_path_noexp)
        datanoexp_path = "./data/dataset_noexp.csv"
        noexp_df = pd.read_csv(datanoexp_path)
        temp_path = "./data/exp/" + "subset_1"
        raw_dataset = load_from_disk(temp_path)
        print("unexplained dataset length is:")
        print(noexp_df.shape[0])
        # embedding process----------------------------------------------
        model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
        tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")

        # use raw_dataset to initial the model_NN
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        # takes in only the tweet and creates an embedding
        def tokenize_noexp_function(examples):
            return tokenizer(
                examples["text"], truncation=True, padding=True, return_tensors="pt"
            )


        torch.cuda.empty_cache()

        # splits the dataset into batches of size 10 and passes them through the tokenizer and pre-trained model.
        print("embedding begin")
        emb = []
        train_dataloader = DataLoader(raw_dataset["train"], batch_size=10)
        # if args.model == "bertie":
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

        # converts the embeddings into a tensor and reshapes them to the correct size
        emb = np.array(emb)
        emb = np.vstack(emb)

        embeddings = torch.tensor(emb)
        print(embeddings.shape)

        # if args.model == "bertie":
        print(embeddings.shape[0]/(36))
        # 785
        total_samples = int(10 * embeddings.shape[0] / (36))
        embeddings = torch.reshape(embeddings, (total_samples, 36 * 3))
        print(embeddings.shape)

        # creates a filename using the passed in arguments
        # and then saves the embedding with this name

        save_filename = (
                "./embeddings/NEW_"
                + "bertie"
                + "_embeddings_"
                + "textattack/bert-base-uncased-MNLI"
                + "_"
                + "subset_1"
                + ".pt"
        )
        print(save_filename)
        save_directory = "./embeddings/NEW_bertie_embeddings_textattack"
        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        torch.save(embeddings, save_filename)

        #classifiy the tweets---------------------------------------------------
        print("classify begin")
        train_dataset, val_dataset, test_dataset, idx = get_datasets()

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
            # unlabel_loader,
            criterion,
            optimizer,
            device,
            writer,
        )

        # calls train to start the training, validating and testing process
        trainer.train(
            int("4"),
            # active_learning_epochs=[10,20,30],
            print_frequency=1,
        )

        writer.close()
        # after initialised the pre-trianed model and classifier we random select the data
        # sampling and explanation generation:
        # ===========================================================================
        if noexp_df.shape[0] >= 10:
            sampled_indices = random.sample(range(noexp_df.shape[0]), 10)
        else:
            sampled_indices = random.sample(range(noexp_df.shape[0]), noexp_df.shape[0])

        sampled_data = [raw_dataset_noexp['train'][i] for i in sampled_indices]
        explanations = []
        no_exp_data = []
        new_data = []
        for data in sampled_data:
            tweet = data["text"]
            label = data["labels"]
            print(tweet)
            no_exp_data.append({"text": tweet, "labels": label})
            for _ in range(2):
                # Use openai model to generate an explanation for the tweet
                explanation = generate_explanations(tweet)
                explanations.append(explanation)
                new_data.append({"text": tweet, "labels": label, "exp_and_td": explanation})
                # Extract remaining 34 explanations from "GPTuseExp.txt"
            with open("GPTuseExp.txt", "r") as file:
                extracted_explanations = file.readlines()[:34]
                explanations.extend(extracted_explanations)
                for extracted_explanation in extracted_explanations:
                    new_data.append({"text": tweet, "labels": label, "exp_and_td": extracted_explanation.strip()})

        # Extend raw_dataset with the new explained data
        # raw_dataset.extend(new_data)
        # Convert new_data to a Dataset object
        # new_dataset = Dataset.from_dict(new_data)
        new_data = {"text": [data["text"] for data in new_data],
                    "labels": [data["labels"] for data in new_data],
                    "exp_and_td": [data["exp_and_td"] for data in new_data]}
        existing_data_path = "./data/dataset_exp_subset_1.csv"
        existing_data_df = pd.read_csv(existing_data_path)
        new_data_df = pd.DataFrame(new_data)
        merged_data_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        merged_data_df.to_csv(existing_data_path, index=False)
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

        # Delete the explained data from raw_dataset_noexp
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
        datanoexp_path = "./data/dataset_noexp.csv"
        noexp_df = pd.read_csv(datanoexp_path)


if __name__ == "__main__":
    main()