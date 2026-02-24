# ML4G
# Graph-Based Recommendation Models

This repository contains two PyTorch implementations for graph-based recommendation:

- **GraphSAGE Recommendation Model**  
  Uses the GraphSAGE approach to learn embeddings for users and items from a user–item bipartite graph.

- **Vectorized PinSAGE Recommendation Model**  
  Uses a vectorized version of PinSAGE with offline neighbor precomputation (via random walks with restart) to generate node embeddings.

## Model Overview

![pinsagearch](https://github.com/user-attachments/assets/76f02815-a167-4895-b9c3-1d213eeb51ad)


The PinSAGE model uses importance-based neighbor sampling and hierarchical convolutions to efficiently learn node embeddings on large graphs.

## Training Dynamics

### MovieLens 100K

<img width="855" height="470" alt="100L" src="https://github.com/user-attachments/assets/1d54e275-685a-4152-9fd8-3276e645f3c8" />


This plot shows the training loss over 10 epochs for GraphSAGE and PinSAGE on the MovieLens 100K dataset.

### MovieLens 1M
<img width="855" height="470" alt="1mL" src="https://github.com/user-attachments/assets/4e5f4854-7ff3-4de3-9746-91afd0ff0ceb" />


This plot shows the training loss over 10 epochs on the MovieLens 1M dataset, where PinSAGE converges to a lower loss than GraphSAGE.

## Requirements

- Python 3.6+
- PyTorch
- NumPy, Pandas, scikit-learn

Install dependencies with:
```bash
pip install torch numpy pandas scikit-learn
