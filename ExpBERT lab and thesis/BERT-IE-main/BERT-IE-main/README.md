# ExpBERT-based Active Learning Lab

This lab focuses on the application of pool-based active learning in ExpBERT-based classification tasks. It includes four basic active learning strategies, along with three alternative annotator simulation processes.

## Quick Start

To run the provided examples, execute the following commands:

```bash
python BALD_MCD.py
python Active_Learning_Strategies_Whole.py --dataset active-dataset --annotator preset --strategy rs
```

## Notes
- **BALD_MCD.py**: For neural networks using dropout, implements BALD sampling through MCD.
- **Available Datasets**: `active-dataset`, `full-data`
- **Annotator Options**: `preset`, `human`, `openai`
- **Active Learning Strategies**:
  - `rs`: Random Sampling
  - `us`: Uncertainty Sampling
  - `ds`: Diversity Sampling
  
## Data

The data is stored in the 'data' folder with a '.tsv' extension.
