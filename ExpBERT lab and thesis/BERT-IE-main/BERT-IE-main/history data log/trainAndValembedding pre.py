import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_from_disk
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing
from torch import nn, multiprocessing

def process_batch(batch, model, tokenizer, tokenize_exp_function):
    with torch.no_grad():
        tokenized_train = tokenize_exp_function(batch, tokenizer)
        model_outputs = model(**tokenized_train)
        embeddings = model_outputs["logits"]
        embeddings = embeddings.cpu().detach().numpy()
        return embeddings


def tokenize_exp_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        examples["exp_and_td"],
        truncation=True,
        padding=True,
        return_tensors="pt",
    )


def main():
    model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-MNLI")
    tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-MNLI")

    # use raw_dataset to initial the model_NN
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    torch.cuda.empty_cache()

    # splits the dataset into batches of size 10 and passes them through the tokenizer and pre-trained model.
    print("embedding begin")
    emb = []
    temp_path = "./data/exp/" + "subset_1"
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

    # converts the embeddings into a tensor and reshapes them to the correct size
    emb = np.array(emb)
    emb = np.vstack(emb)

    embeddings = torch.tensor(emb)
    print(embeddings.shape)

    # if args.model == "bertie":
    print(embeddings.shape[0] / (36))
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

if __name__ == "__main__":
    main()