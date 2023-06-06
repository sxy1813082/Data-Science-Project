import torch

# the filename is changed depending on which embeddings are being concatenated
filename = "expbert_embeddings_distilroberta-base"

embeddings_1 = torch.load("./embeddings/" + filename + "_subset_1.pt")
embeddings_2 = torch.load("./embeddings/" + filename + "_subset_2.pt")
embeddings_3 = torch.load("./embeddings/" + filename + "_subset_3.pt")
embeddings_4 = torch.load("./embeddings/" + filename + "_subset_4.pt")
embeddings_5 = torch.load("./embeddings/" + filename + "_subset_5.pt")
embeddings_6 = torch.load("./embeddings/" + filename + "_subset_6.pt")
embeddings_7 = torch.load("./embeddings/" + filename + "_subset_7.pt")
embeddings_8 = torch.load("./embeddings/" + filename + "_subset_8.pt")
embeddings_9 = torch.load("./embeddings/" + filename + "_subset_9.pt")

concatenated = torch.cat(
    (
        embeddings_1,
        embeddings_2,
        embeddings_3,
        embeddings_4,
        embeddings_5,
        embeddings_6,
        embeddings_7,
        embeddings_8,
        embeddings_9,
    ),
    0,
)
print(type(concatenated))
print(concatenated.shape)
print(concatenated[0].shape)

torch.save(concatenated, f"./embeddings/" + filename + "_combined.pt")
