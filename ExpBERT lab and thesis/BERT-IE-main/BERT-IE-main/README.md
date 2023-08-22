## The lab is for applying pool-based active learning in the ExpBERT-based classification tasks. Include four basic active learning strategies with three alternative annotator simulation processes.

#### Please run the example.sh script:

python BALD_MCD.py

python Active Learning Strategies Whole.py --dataset active-dataset --annotator preset --strategy rs

## PS:
##### BALD_MCD.py is for dropout NN use BALD sampling through the MCD
##### The dataset to run: active-dataset, full-data
##### annotator type: preset, human, openai
##### active learning strategies type: rs (random sampling), us (uncertainty sampling), ds (diversity sampling)


