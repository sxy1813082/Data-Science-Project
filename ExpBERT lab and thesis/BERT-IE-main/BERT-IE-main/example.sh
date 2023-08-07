# example bash script for classifying the dataset using the BERT-IE method
#python prepare_dataset_all.py
#python prepare_embeddings_all.py --pretrained textattack/bert-base-uncased-MNLI --model bertie --subset subset_1
#python classify_tweets_all.py --embedding bertie_embeddings_nli-bert_all.pt --model bertie --lr 5e-5 --run 1

# dataset have whole dataset 'Whole-dataset' and active learning dataset 'active-dataset'
# Whole_train.py is for random sampling
python Whole_train.py --dataset active-dataset

# the process of add explanation have 3 types
#   1. use pre set explanation 'pre-set'  2. use human input 'human-set' 3.use OpenAI model 'openai-set'
# diversity_sampling_whole.py is for diversity sampling
python diversity_sampling_whole.py --exp pre-set

# uncertaintyWhole.py is for uncertainty sampling
python uncertaintyWhole.py

# BALD_MCD.py is for dropout NN use BALD sampling
python BALD_MCD.py

