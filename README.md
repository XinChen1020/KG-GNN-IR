# KG-GNN-IR



Complex questions requiring multihop reasoning pose distinct challenges in information retrieval. This capstone project explores the use of Graph Neural Networks (GNN) to potentially enhance retrieval strategies for such queries. Central to our approach is the construction of a knowledge graph that organizes information by linking passages to extracted entities and the titles of their source articles. This structuring allows the GNN to leverage the relational data between entities, aiding in the exploration of more effective retrieval strategies. The project is focused on developing and testing this framework to examine how GNNs can be integrated with knowledge graphs to assist in handling complex informational queries.

## Getting Started

These instructions will get your copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before running the scripts, you'll need to install the required Python packages. You can install all the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Installing and Running

Follow these steps to get a development environment running:

## Download HotpotQA Dataset
Run the following bash command to download the required HotpotQA dataset:
```bash
./dataset/download_datasets.sh
```

## Run Baseline Model 
To execute the baseline model with default settings, use:
```bash
python hotpotqa_baseline.py
```
You can customize the script's execution by adjusting the command-line parameters:
```bash
python hotpotqa_baseline.py --model_name "YourModelName" --file_name "your_file.json"
```
## Build the Knowledge Graph
To construct the knowledge graph from the HotpotQA training file, execute:
```bash
python GraphBuilder.py
```
This will output a JSON file containing the triplets in kgs.json.

## Training the Model
To train the model, run:
```bash
python train.py
```
The model achieving the highest hit rate will be automatically stored in the output folder.

## Built With
PyTorch - An open source machine learning framework.
PyTorch Geometric - A library for deep learning on graph and other irregular structures.
LlamaIndex - Custom library for indexing and processing text data.
