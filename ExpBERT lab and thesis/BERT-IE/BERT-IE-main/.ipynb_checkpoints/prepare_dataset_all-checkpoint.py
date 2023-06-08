import os
import pandas as pd
import emoji
import numpy as np
from datasets import load_dataset


# retrieves all the filenames in the directory that end with .tsv
# as these are the raw datasets
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
    df_noexp = df_concat.drop_duplicates(subset=["text"], inplace=False)

    """ code only for NoExp """
    # saves the dataset to a dataset directory
    df_noexp.to_csv("./data/dataset_noexp.csv", index=False)
    data_noexp = load_dataset("csv", data_files="./data/dataset_noexp.csv")
    data_noexp.save_to_disk("./data/")

    """ code only for ExpBERT and BERT-IE """
    # reads in explanations and concatenates to the tweets to form an expanded dataset
    explanations = read_explanations("explanations.txt")
    df_exp = create_explanations_dataset(df_noexp, explanations)

    # splits the expanded dataset into nine subsets
    # so it can be passed through the pre-trained model
    # each subset is created and then saved
    subset_1 = df_exp[0:72000]
    subset_1.to_csv("./data/dataset_exp_subset_1.csv", index=False)
    subset_1_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_1.csv")
    subset_1_dict.save_to_disk("./data/exp/subset_1")

    subset_2 = df_exp[72000:144000]
    subset_2.to_csv("./data/dataset_exp_subset_2.csv", index=False)
    subset_2_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_2.csv")
    subset_2_dict.save_to_disk("./data/exp/subset_2")

    subset_3 = df_exp[144000:216000]
    subset_3.to_csv("./data/dataset_exp_subset_3.csv", index=False)
    subset_3_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_3.csv")
    subset_3_dict.save_to_disk("./data/exp/subset_3")

    subset_4 = df_exp[216000:288000]
    subset_4.to_csv("./data/dataset_exp_subset_4.csv", index=False)
    subset_4_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_4.csv")
    subset_4_dict.save_to_disk("./data/exp/subset_4")

    subset_5 = df_exp[288000:360000]
    subset_5.to_csv("./data/dataset_exp_subset_5.csv", index=False)
    subset_5_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_5.csv")
    subset_5_dict.save_to_disk("./data/exp/subset_5")

    subset_6 = df_exp[360000:432000]
    subset_6.to_csv("./data/dataset_exp_subset_6.csv", index=False)
    subset_6_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_6.csv")
    subset_6_dict.save_to_disk("./data/exp/subset_6")

    subset_7 = df_exp[432000:504000]
    subset_7.to_csv("./data/dataset_exp_subset_7.csv", index=False)
    subset_7_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_7.csv")
    subset_7_dict.save_to_disk("./data/exp/subset_7")

    subset_8 = df_exp[504000:576000]
    subset_8.to_csv("./data/dataset_exp_subset_8.csv", index=False)
    subset_8_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_8.csv")
    subset_8_dict.save_to_disk("./data/exp/subset_8")

    subset_9 = df_exp[576000:616212]
    subset_9.to_csv("./data/dataset_exp_subset_9.csv", index=False)
    subset_9_dict = load_dataset("csv", data_files="./data/dataset_exp_subset_9.csv")
    subset_9_dict.save_to_disk("./data/exp/subset_9")
    """ """


if __name__ == "__main__":
    main()
