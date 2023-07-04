import os
import pandas as pd
import emoji
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing

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

def tokenize_exp_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        examples["exp_and_td"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

def preprocess_samples(raw_dataset_noexp):
    print("preprocess_samples begin ...")
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

    total_samples = int(10 * embeddings.shape[0] / (18))
    embeddings = torch.reshape(embeddings, (total_samples, 18 * 3))
    print("uncertainty shape:", embeddings.shape[0])
    idx = np.arange(0, len(embeddings), dtype=np.intc)
    embeddings = embeddings[idx]
    return embeddings

def main():
    # test dataset
    test_dataframes = []
    explanations = read_explanations("explanations.txt")
    filepaths = obtain_filepaths("./test_data/")
    # cleans the data from each disaster individually
    for file in filepaths:
        df = clean_individual_dataset(file)
        test_dataframes.append(df)
    # concatenates the tweets from each disaster to form one dataset
    test_df_concat = pd.concat(test_dataframes)
    # renames the columns
    test_df_concat.rename(columns={"tweet_text": "text"}, inplace=True)
    test_df_concat.rename(columns={"label": "labels"}, inplace=True)
    # duplicate tweets are dropped
    test_noexp_all = test_df_concat.drop_duplicates(subset=["text"], inplace=False)
    test_noexp_all.to_csv("./test_data/dataset_noexp.csv", index=False)
    data_noexp = load_dataset("csv", data_files="./test_data/dataset_noexp.csv")
    data_noexp.save_to_disk("./test_data/")

    test_exp = create_explanations_dataset(test_noexp_all, explanations)
    print("test len", len(test_exp))
    subset_test = test_exp[0:(len(test_exp) // 180) * 180]
    subset_test.to_csv("./data/dataset_exp_subset_test.csv", index=False)
    subset_test_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_test.csv")
    subset_test_dict.save_to_disk("./data/exp/subset_test")
    test_raw_dataset = load_from_disk("./data/exp/subset_test")
    test_embedding = preprocess_samples(test_raw_dataset)
    save_filename = (
            "./test_embeddings/NEW_"
            + "bertie"
            + "_embeddings_"
            + "textattack/bert-base-uncased-MNLI"
            + "_"
            + "subset_1"
            + ".pt"
    )
    print(save_filename)
    save_directory = "./test_embeddings/NEW_bertie_embeddings_textattack"
    # Create the directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    torch.save(test_embedding, save_filename)

if __name__ == "__main__":
    main()
