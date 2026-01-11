# Data Science Homework
A collection of homework assignments for the Data Science course at National Tsing Hua University (NTHU). This repository contains implementations for frequent pattern mining, graph neural network-based node classification, and global optimization algorithms.

---

## Table of Contents
- [HW1: Frequent Pattern Mining](#hw1-frequent-pattern-mining)
- [HW2: Node Classification on Graph Data](#hw2-node-classification-on-graph-data)
- [HW3: Global Optimization](#hw3-global-optimization)
- [HW4: Model Compression](#hw4-model-compression)
- [HW5: Horizontal Federated Learning](#hw5-horizontal-federated-learning)

---

## HW1: Frequent Pattern Mining

### Overview
Given a set of transactions and a minimum support threshold, implement an algorithm to find all frequent patterns. The algorithm choice is flexible—Apriori, FP-Growth, or other approaches are all acceptable.

### Task Description
- **Input**: A text file containing transactions, where each line represents a transaction with items separated by commas (no spaces)
- **Output**: A text file listing all frequent patterns with their support values
- **Constraints**:
  - Items are integers in range 0–999
  - Up to 100,000 transactions
  - Up to 200 items per transaction
  - No frequent pattern mining libraries allowed (e.g., `apyori`, `pyfpgrowth`)

### Input Format
```
5,9,10
0,1,4,6,8,10
0,1,10
5
0,1,3,8,10
...
```

### Output Format
```
pattern:support
```
Example:
```
2,9:0.2500
4,9:0.2500
0,1,10:0.2500
```
- Support values are rounded to 4 decimal places

### How to Run

**Python:**
```bash
python3 {student_id}_hw1.py [min_support] [input_file] [output_file]
```

**C++:**
```bash
# Compile
g++ -std=c++2a -pthread -fopenmp -O2 -o {student_id}_hw1 {student_id}_hw1.cpp

# Run
./{student_id}_hw1 [min_support] [input_file] [output_file]
```

**Example:**
```bash
python3 112345_hw1.py 0.2 input1.txt output1.txt
```

---

## HW2: Node Classification on Graph Data

### Overview
Build a Graph Neural Network (GNN) model for semi-supervised node classification. Given a graph dataset with limited labeled nodes, train a model to predict labels for testing nodes.

### Task Description
- **Objective**: Classify nodes in a graph using semi-supervised learning
- **Challenge**: Training data is significantly smaller than validation/testing data (60 training vs 600 validation vs 1200 testing nodes)
- **Framework**: DGL (Deep Graph Library) with PyTorch

### Dataset Structure
```
dataset/
├── private_features.pkl      # Node features
├── private_graph.pkl         # Graph edges
├── private_num_classes.pkl   # Number of classes
├── private_train_labels.pkl  # Training node labels
├── private_train_mask.pkl    # Training node indices
├── private_val_labels.pkl    # Validation node labels
├── private_val_mask.pkl      # Validation node indices
├── private_test_labels.pkl   # Test labels (placeholder)
└── private_test_mask.pkl     # Test node indices
```

### Output Format
CSV file with header `ID,Predict`:
```csv
ID,Predict
0,1
1,0
2,0
3,2
...
```

### How to Run
```bash
python3 train.py --epochs 300 --es_iters 30 --use-gpu
```

**Arguments:**
- `--epochs`: Number of training epochs
- `--es_iters`: Early stopping patience
- `--use-gpu`: Enable GPU acceleration

### Project Structure
```
├── data_loader.py    # Data loading utilities (do not modify)
├── model.py          # GNN model definition (implement your model here)
└── train.py          # Training and evaluation script
```

### Baseline Model
The provided baseline uses a 2-layer GCN (Graph Convolutional Network):
```python
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
```



### References
- [DGL Documentation](https://docs.dgl.ai/index.htm)
- [GAT Paper](https://arxiv.org/pdf/1710.10903v3.pdf)
- [SSP Paper](https://arxiv.org/pdf/2008.09624v1.pdf)
- [GRACE Paper](https://arxiv.org/abs/2006.04131v2)
- [State-of-the-art Node Classification](https://paperswithcode.com/task/node-classification)

---

## HW3: Global Optimization

### Overview
Design an optimization algorithm to find the global minimum of unknown black-box functions with limited function evaluations.

### Task Description
- **Objective**: Find the global minimum of 4 unknown objective functions
- **Constraint**: Limited number of function evaluations per function
- **Approach**: Implement metaheuristic optimization algorithms (e.g., CMA-ES, Differential Evolution, EDA)

### Function API
The optimizer must inherit from the `Function` class and use:
```python
self.f.dimension(func_num)    # Get input dimension
self.f.upper(func_num)        # Get upper bound
self.f.lower(func_num)        # Get lower bound
self.f.evaluate(func_num, x)  # Evaluate function at point x
```

### Function Evaluation Limits
| Function | Max Evaluations |
|----------|-----------------|
| Function 1 | 1000 |
| Function 2 | 1500 |
| Function 3 | 2000 |
| Function 4 | 2500 |

### Baseline Results
**Public Functions:**
| Function | Random Search | Better Baseline |
|----------|---------------|-----------------|
| 1 | 0.036 | 1.875e-6 |
| 2 | 0.381 | 4.042e-9 |
| 3 | 13.427 | 0.210 |
| 4 | 67.743 | 0.530 |

**Private Functions:**
| Function | Random Search | Better Baseline |
|----------|---------------|-----------------|
| 1 | 19.685 | 0.412 |
| 2 | 15.215 | 8.066 |
| 3 | 2246.339 | 1620.202 |
| 4 | -4.155 | -7.738 |

### Output Format
Four text files, one per function:
```
{student_id}_function1.txt
{student_id}_function2.txt
{student_id}_function3.txt
{student_id}_function4.txt
```

Each file contains the best input parameters followed by the objective value:
```
1.234
5.678
...
-0.5432
```

### How to Run
```bash
python3 {student_id}_hw3.py
```

### Constraints
- **Time limit**: 5 min/function | **Libraries**: NumPy only
- **Environment**: Ubuntu 20.04.6, Python 3.10.17, NumPy 1.24.4

### References
- [CMA-ES](https://arxiv.org/abs/1604.00772) | [CoDE](https://ieeexplore.ieee.org/document/5688232) | [EDA/LS](https://ieeexplore.ieee.org/document/7001197)

---

## HW4: Model Compression

### Overview
Implement Deep Compression techniques for neural network model compression. The pipeline includes Pruning, Quantization, and Huffman Coding to reduce model storage while maintaining acceptable accuracy. Base model: AlexNet on CIFAR-10.

### Task Description
- **Objective**: Compress a deep neural network using Deep Compression pipeline
- **Focus**: Pruning and Huffman Coding (Quantization is optional/simplified)
- **Goal**: Analyze trade-off between compression rate and classification accuracy

### Implementation Requirements

**1. Pruning (`prune.py`, `pruning.py`)**
- Implement standard deviation-based pruning for fully-connected layers
- Optional: percentile-based pruning for predictable compression rate
- Includes initial training and prune-retrain workflow

**2. Quantization (`quantization.py`)**
- Implement weight sharing using K-means clustering
- Apply only to fully-connected layers

**3. Huffman Coding (`huffmancoding.py`)**
- Implement Huffman encoding/decoding
- Compress weights of fully-connected layers only

### How to Run
```bash
# Step 1: Train and prune
python pruning.py

# Step 2: Quantization (optional)
python quantization.py

# Step 3: Huffman coding
python huffmancoding.py
```


---

## HW5: Horizontal Federated Learning

### Overview
Implement the core components of Horizontal Federated Learning (HFL) using the FedAvg algorithm. Train a global model across multiple participants without sharing raw data.

### Task Description
- **Objective**: Implement server-side aggregation and user-side parameter synchronization for federated learning
- **Dataset**: CIFAR-10 with Dirichlet distribution for non-IID data splitting
- **Model**: ResNet-18

### Implementation Requirements

**Server (`serverbase.py`)** - Implement 2 functions:
- `select_users(self, model, beta)`: Select which users participate in each training round
- `aggregate_parameters(self)`: Weighted average of user models based on data proportion to form global model

**User (`userbase.py`)** - Implement 1 function:
- `set_parameters(self, mode, beta)`: Initialize local model with global model parameters

### Project Structure
```
HFL/
├── data/CIFAR10/
│   └── generate_niid_dirichlet.py    # Data splitting script
├── FLAlgorithms/
│   ├── servers/
│   │   ├── serverbase.py             # TODO: implement here
│   │   └── serveravg.py
│   ├── trainmodel/
│   │   └── resnet.py
│   └── users/
│       ├── userbase.py               # TODO: implement here
│       └── useravg.py
├── utils/
├── main.py
└── run.sh
```

### How to Run

**1. Generate non-IID data:**
```bash
cd ./data/CIFAR10
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 1.0 --alpha 50.0 --n_user 10
```

**2. Train model:**
```bash
python main.py --dataset CIFAR10-alpha50.0-ratio1.0-users10 --algorithm FedAvg \
    --num_glob_iters 150 --local_epochs 10 --num_users 10 \
    --learning_rate 0.1 --model resnet18 --device cuda
```

### Experiments to Explore
1. **Data Distribution**: Compare alpha ∈ {0.1, 50.0} - analyze user data distribution and global model accuracy
2. **Number of Users**: Compare num_users ∈ {2, 10} - analyze accuracy and convergence speed


