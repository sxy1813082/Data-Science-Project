# example bash script for classifying the dataset using the BERT-IE method
python prepare_dataset_all.py
python prepare_embeddings_all.py --pretrained textattack/bert-base-uncased-MNLI --model bertie --subset subset_1
python classify_tweets_all.py --embedding bertie_embeddings_nli-bert_all.pt --model bertie --lr 5e-5 --run 1