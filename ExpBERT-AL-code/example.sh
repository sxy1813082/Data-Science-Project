# example bash script for classifying
# the dataset to run: active-dataset (with 15% training dataset), full-data (with 70% training dataset)
# annotator type: preset, human, openai
# active learning strategies type: rs (random sampling), us (uncertainty sampling), ds (diversity sampling)
python Active_Learning_Strategies_Whole.py --dataset active-dataset --annotator preset --strategy rs

# BALD_MCD.py is for dropout NN use BALD sampling through the MCD
python BALD_MCD.py




