import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter

import argparse

from tqdm.auto import tqdm

from datasets import load_from_disk

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from typing import Callable

from math import floor

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
torch.multiprocessing.set_sharing_strategy("file_system")

# Passing arguments in via the command line --------------------
parser = argparse.ArgumentParser()

parser.add_argument(
    "--embedding", type=str, required=True, help="the file path for the embeddings"
)
parser.add_argument(
    "--epochs",
    type=str,
    required=False,
    default="40",
    help="number of epochs to train the model head for",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    default="noexp",
    help="Model type: noexp, expbert, bertie",
)
parser.add_argument(
    "--subset_size",
    type=str,
    required=False,
    default="72000",
    help="how big is the subset of data",
)
parser.add_argument(
    "--hidden_layer_size",
    type=str,
    required=False,
    default="100",
    help="how many neurons in the hidden layer",
)
parser.add_argument(
    "--lr", type=str, required=False, default="1e-3", help="learning rate of the model"
)
parser.add_argument(
    "--wd",
    type=str,
    required=False,
    default="1e-2",
    help="weight decay of the optimiser",
)
parser.add_argument("--run", type=str, required=True, help="what run number this is")
parser.add_argument(
    "--seed", type=str, required=False, default="37", help="seed for consistent results"
)
parser.add_argument(
    "--WCEL", type=str, required=False, default="False", help="WCEL or CEL"
)
parser.add_argument(
    "--other", type=str, required=False, default="other", help="space for extra info"
)

args = parser.parse_args()

# Setting up the tensorboard for visualising results --------------------
tensorboard_filepath = (
    args.model
    + "_"
    + args.epochs
    + "_"
    + args.lr
    + "_wd:"
    + args.wd
    + "_run:"
    + args.run
    + "_seed:"
    + args.seed
    + "_other:"
    + args.other
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


# Retrieves the embeddings from the given file path and splits dataset into training, validation and test --------------------
def get_datasets():
    with torch.no_grad():
        # loading in the embeddings
        file_path = "./embeddings/" + args.embedding

        embeddings = torch.load(file_path)
        raw_dataset = load_from_disk("./data/")
        labels = np.array(raw_dataset["train"]["labels"])

        # shuffling embeddings and labels
        idx = np.arange(0, len(embeddings), dtype=np.intc)
        np.random.seed(int(args.seed))
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
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.writer = writer

    # training the model
    def train(self, epochs: int, print_frequency: int = 20, start_epoch: int = 0):
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

                loss = self.criterion(logits, labels)
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
                loss = self.criterion(logits, labels)
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
                loss = self.criterion(logits, labels)
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


def main():
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

    torch.manual_seed(args.seed)

    # initialises the model and hyperparameters taking into account the passed in arguments
    model = MLP_1h(feature_count, int(args.hidden_layer_size), class_count)

    model = model.to(device)
    optimizer = AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.wd)
    )

    if args.WCEL == "True":
        weights = get_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # initalises the Trainer class
    trainer = Trainer(
        model,
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
        int(args.epochs),
        print_frequency=1,
    )

    writer.close()


if __name__ == "__main__":
    main()
