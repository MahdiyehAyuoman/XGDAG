# XGDAG: Code and Evaluation Summary
## Comprehensive Technical Reference for Scientific Article Use

> **Repository:** [MahdiyehAyuoman/XGDAG](https://github.com/MahdiyehAyuoman/XGDAG)  
> **Original paper:** Mastropietro, A., De Carlo, G., & Anagnostopoulos, A. (2023). XGDAG: explainable gene–disease associations via graph neural networks. *Bioinformatics*, 39(8), btad482. https://doi.org/10.1093/bioinformatics/btad482

---

## 1. Overview

XGDAG (**eX**plainable **G**ene–**D**isease **A**ssociations via **G**raph neural networks) is a framework for identifying candidate disease-associated genes in the human Protein–Protein Interaction (PPI) network by combining Graph Neural Network (GNN) classification with post-hoc graph explainability methods. Given a disease of interest, the pipeline: (1) constructs a disease-specific subgraph of the PPI network, (2) trains a node-classification GNN to distinguish known disease genes from other genes, and (3) uses GNN explainability methods to rank previously unknown genes as novel disease gene candidates.

---

## 2. Repository Structure

| File / Folder | Purpose |
|---|---|
| `CreateGraph.py` | Builds NetworkX graph from BioGRID PPI data; attaches NeDBIT node features |
| `CreateDataset.py` / `CreateDatasetv2.py` | Converts graph + gene-disease labels into PyTorch Geometric `InMemoryDataset` |
| `CreateDatasetv2_binary.py` / `CreateDatasetv2_binary_diamond.py` | Same as above but with binary (Positive / Unlabelled) labeling |
| `GraphSageModel.py` | Defines the 7-layer GraphSAGE GNN architecture |
| `GNNTrain.py` | Training loop, model checkpointing, classification reports, and confusion matrices |
| `GDARanking.py` | Core explainability-based gene ranking (GNNExplainer, GraphSVX, SubgraphX) |
| `ComputeRankingScript.py` | CLI entry-point for end-to-end ranking computation |
| `TrainerScript.py` | CLI entry-point for GNN training |
| `configs.py` | Argument parser and default hyper-parameters for training and evaluation |
| `Paths.py` | Central path constants for datasets, graphs, models, rankings, and metrics |
| `NetworkDataAnalysis.ipynb` | Exploratory data analysis of the PPI and PPI+GDA networks |
| `comparison_plots_disgenet.ipynb` | Evaluation and plots comparing methods on DisGeNET validation set |
| `comparison_plots_omim.ipynb` | Evaluation and plots comparing methods on OMIM+PheI validation set |
| `comparison_plots_omim_vs_disgenet.ipynb` | Cross-dataset comparison plots |
| `ModelsPerformancesComparison.ipynb` | Comparison of GNN classification performance across diseases |
| `Datasets/` | Seed gene lists, NeDBIT feature files, and gene rankings per disease |
| `Graphs/` | Pre-built GML graph files (PPI + disease-specific subgraphs) |
| `Models/` | Pre-trained GraphSAGE model checkpoints |
| `Rankings/` | Computed gene rankings (XGDAG + baseline methods: GUILD NetScore, etc.) |
| `Metrics/` | Pickled evaluation metric dictionaries |

---

## 3. Data Sources

### 3.1 PPI Network — BioGRID
The human PPI network is constructed from **BioGRID** interaction data. During preprocessing:
- Only human–human interactions are retained (Organism ID = 9606).
- Self-loops are removed.
- Only the **Largest Connected Component (LCC)** is kept.

The resulting PPI network has the following properties (from `NetworksEDA.txt`):
- **Nodes:** 19,761 proteins  
- **Edges:** 678,932 interactions  
- **Average degree:** 68.7  
- **Diameter:** 7  
- **Average shortest path length:** 2.80  
- **Clustering coefficient:** 0.115

### 3.2 Gene–Disease Associations — DisGeNET
Disease seed genes (known disease-associated genes) are sourced from **DisGeNET**. For each disease, a file `<disease_id>_seed_genes.txt` lists the confirmed seed genes along with their GDA (Gene–Disease Association) scores. Ten diseases are included, identified by UMLS Concept IDs (e.g., `C0006142` for Breast Cancer, `C0009402` for Colorectal Cancer, `C0036341` for Schizophrenia, etc.).

### 3.3 Network Expansion — DIAMOnD
The `Diamond_dataset/` subfolder contains results of the **DIAMOnD** algorithm applied to the PPI network. DIAMOnD (DIseAse MOdule Detection) iteratively adds nodes most connected to the current disease module, expanding the set of seed genes into a larger candidate module. The expanded ranked list (`<disease_id>_ranking`) and expanded seed files are used as the primary data for GNN training and evaluation (the `diamond` variants of graphs and datasets).

### 3.4 NeDBIT Node Features
Each node in the graph is annotated with six network-based features derived from the **NeDBIT** (Network-Based Integration of omics data for diseases and BIological processes, Top-scoring) tool:

| Feature | Description |
|---|---|
| `degree` | Node degree in the PPI |
| `ring` | Clustering ring score |
| `NetRank` | Global network centrality rank |
| `NetShort` | Shortest-path–based network proximity score |
| `HeatDiff` | Heat diffusion–based propagation score |
| `InfoDiff` | Information diffusion–based propagation score |

Features are stored in `Datasets/<disease_id>_features`. Optionally, they are normalized with `RobustScaler` before training.

---

## 4. Graph Construction (`CreateGraph.py`)

```
BioGRID TSV  ──►  NetworkX Graph  ──►  Remove self-loops  ──►  Extract LCC
                                                                     │
                                                          Attach NeDBIT features
                                                                     │
                                                          Save as .gml file
```

The `create_graph_from_PPI()` function:
1. Reads BioGRID interactions and filters to human proteins.
2. Creates an undirected `networkx.Graph` where nodes are official gene symbols.
3. Removes self-loops.
4. Extracts the LCC, ensuring graph connectivity.
5. Loads NeDBIT feature files and attaches six scalar features to every node.
6. Optionally scales features with `RobustScaler`.
7. Saves the graph in GML format for reuse.

---

## 5. Dataset Construction and Node Labeling

### 5.1 Multiclass Labeling (`CreateDatasetv2.py`)

Each node in the graph is assigned one of five labels derived from its position relative to the known disease module:

| Label | Code | Meaning |
|---|---|---|
| **P** (Positive) | 0 | Confirmed seed gene from DisGeNET / DIAMOnD |
| **LP** (Likely Positive) | 1 | Top quartile of NeDBIT score among non-seed genes |
| **WN** (Weakly Negative) | 2 | Second quartile |
| **LN** (Likely Negative) | 3 | Third quartile |
| **RN** (Random Negative) | 4 | Bottom quartile |

Non-seed genes are ranked by their NeDBIT score and divided into four equal quartiles using `pd.qcut`. Seed genes are unconditionally assigned label **P**. This semi-supervised labeling strategy transforms the unsupervised network into a labeled dataset without requiring manual annotation beyond the known seed genes.

### 5.2 Binary Labeling (`CreateDatasetv2_binary_diamond.py`)

A simplified version uses two classes:
- **P** (0): Confirmed seed gene
- **U** (1): All other genes (Unlabelled)

### 5.3 Train / Validation / Test Split

Nodes are split 70 / 15 / 15 using **stratified** `train_test_split` (scikit-learn) with `random_state=42`, ensuring class proportions are preserved across splits. Boolean masks (`train_mask`, `val_mask`, `test_mask`) are stored in the `PyTorch Geometric` Data object.

---

## 6. GNN Model Architecture (`GraphSageModel.py`)

XGDAG uses a **7-layer GraphSAGE** (Graph Sample and Aggregate) network:

```
Input (6 NeDBIT features)
       │
  SAGEConv(6 → 16, aggr='max')  + ReLU
  SAGEConv(16 → 16, aggr='max') + ReLU
  SAGEConv(16 → 16, aggr='max') + ReLU
  SAGEConv(16 → 16, aggr='max') + ReLU
  SAGEConv(16 → 16, aggr='max') + ReLU
  SAGEConv(16 → 16, aggr='max') + ReLU  + Dropout
  SAGEConv(16 → num_classes, aggr='max')
       │
  log_softmax
       │
  Output (5 classes or 2 classes)
```

- **Aggregation:** max-pooling over neighborhood
- **Activation:** ReLU after each layer; dropout before the last layer
- **Output:** log-softmax probabilities per class

The depth of 7 layers allows each node to aggregate information from a 7-hop neighborhood, capturing both local and global network topology around each gene.

---

## 7. Training (`GNNTrain.py`, `TrainerScript.py`)

Default hyper-parameters:
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=0.0005`)
- **Loss:** Negative Log-Likelihood (NLL) loss
- **Epochs:** 40,000
- **Model selection:** checkpoint saved at epoch with lowest training loss

The `train()` function logs training accuracy and loss every 20 epochs and generates:
- Training accuracy / loss curves
- Classification report (precision, recall, F1 per class)
- Confusion matrices (absolute and row-normalized)

Pre-trained checkpoints for all 10 diseases are provided in the `Models/` folder.

---

## 8. Gene Ranking via Explainability (`GDARanking.py`)

After training, the GNN is used not just as a classifier but as a **scoring mechanism** for candidate gene discovery. The central idea is: if a gene appears consistently in the explanatory subgraphs of many confirmed disease genes, it is likely a disease-associated candidate itself.

### 8.1 Core Ranking Procedure (All Methods)

For each confirmed **seed gene (P)** in the graph:
1. Compute the explainability mask/scores over its local neighborhood.
2. Identify neighbor genes with high importance scores that are also classified as **LP** (Likely Positive).
3. Aggregate: for each candidate gene, record how many seed-gene explanations include it and the cumulative importance score.

Final ranking is sorted by: (1) number of seed-gene explanations the candidate appears in, then (2) cumulative importance score.

### 8.2 Supported Explainability Methods

| Method | Implementation | Description |
|---|---|---|
| `gnnexplainer` | GNNExplainer (PyTorch Geometric) | Learns soft edge masks per node; averaged over 10 runs to reduce variance |
| `gnnexplainer_only` | GNNExplainer | Same but considers all unlabelled neighbors (not just LP) |
| `graphsvx` | GraphSVX | Shapley-value–based explanations using coalition sampling (100 coalitions) |
| `graphsvx_only` | GraphSVX | Shapley-value variant on unlabelled genes with mean-threshold filtering |
| `subgraphx` | SubgraphX | Monte Carlo tree search over subgraphs; parallelized via `multiprocessing.Pool`; degree filter ≤ 20 applied to seed genes |
| `subgraphx_only` | SubgraphX (binary) | Same but for 2-class model |

All methods are invoked through the unified `predict_candidate_genes()` dispatcher.

### 8.3 Baseline Methods for Comparison

Pre-computed rankings from the following methods are stored in `Rankings/other_methods/`:
- **GUILD** (NetScore algorithm): network-based prioritization using graph diffusion
- Additional baselines from the original paper

---

## 9. Evaluation and Validation

### 9.1 Validation Datasets

Two independent validation datasets are used:
1. **DisGeNET holdout set** (`comparison_plots_disgenet.ipynb`): Disease-gene associations from DisGeNET not used as seeds.
2. **OMIM + PheI** (`comparison_plots_omim.ipynb`): Gene–disease associations from OMIM and PhenolyzerDB databases.

### 9.2 Evaluation Metrics

Rankings are evaluated by measuring what fraction of the top-k ranked genes are known disease genes from the validation set. Typical metrics include:
- **Precision@k** — fraction of top-k candidates confirmed by the validation set
- **Recall@k** — fraction of validation-set genes recovered in the top-k ranking
- **AUC / AUROC** — area under the ROC curve for ranking quality

Metrics are computed for each disease and method, stored as pickled dictionaries in `Metrics/` (`disease_method_metrics.pickle`, `diamond_disease_method_metrics.pickle`).

### 9.3 Network Analysis (`NetworkDataAnalysis.ipynb`)

For each disease, the following network topology statistics are computed on the PPI+GDA subgraph (LCC of PPI restricted to disease-associated genes):
- Number of nodes and edges
- Average degree
- Connected components
- Diameter and radius
- Clustering coefficient
- Density
- Average shortest path length
- Betweenness, closeness, and eigenvector centrality

---

## 10. Studied Diseases

| UMLS ID | Disease |
|---|---|
| C0001973 | Alcoholic Liver Disease |
| C0005586 | Bipolar Disorder |
| C0006142 | Breast Cancer |
| C0009402 | Colorectal Cancer |
| C0011581 | Depressive Disorder |
| C0023893 | Liver Cirrhosis |
| C0036341 | Schizophrenia |
| C0376358 | Prostate Cancer |
| C0860207 | Drug-Induced Liver Disease |
| C3714756 | Autism Spectrum Disorder |

---

## 11. Reproducibility Notes

- All random seeds are fixed at 42: `torch.manual_seed(42)` in `CreateDataset.py` / `CreateDatasetv2.py`; `torch.manual_seed(SEED)`, `np.random.seed(SEED)`, `random.seed(SEED)` in `GDARanking.py`; `train_test_split(..., random_state=42)` in all dataset modules.
- Pre-trained models are provided for all 10 diseases; training from scratch is optional.
- All pre-computed rankings (XGDAG variants and baselines) are included in `Rankings/`.
- Environment specifications (Conda YAML files) are in `CondaEnvs/`.
- Requires **PyTorch 1.12** and **PyTorch Geometric 2.1**.

---

## 12. How to Cite

If you use this code or data in your research, please cite:

```bibtex
@article{mastropietro2023xgdag,
  author  = {Mastropietro, Andrea and De Carlo, Gianluca and Anagnostopoulos, Aris},
  title   = {{XGDAG}: explainable gene–disease associations via graph neural networks},
  journal = {Bioinformatics},
  volume  = {39},
  number  = {8},
  pages   = {btad482},
  year    = {2023},
  doi     = {10.1093/bioinformatics/btad482}
}
```

---

## 13. Summary for Scientific Article Methods Section

The following paragraph summarizes the pipeline in a style suitable for a Methods section in a bioinformatics article:

> We utilized the XGDAG framework (Mastropietro et al., 2023) for disease gene ranking in the human PPI network. The PPI network was constructed from BioGRID interaction data, retaining only human–human interactions and the largest connected component (19,761 proteins; 678,932 interactions). Known disease-associated seed genes were obtained from DisGeNET and expanded using the DIAMOnD algorithm. Each protein node was annotated with six NeDBIT network-propagation features (degree, ring coefficient, NetRank, NetShort, HeatDiff, InfoDiff). Nodes were assigned pseudo-labels (Positive, Likely Positive, Weakly Negative, Likely Negative, Random Negative) based on their NeDBIT scores relative to seed genes, creating a five-class semi-supervised learning setting. A seven-layer GraphSAGE model (max-pooling aggregation) was trained on this labeled graph to classify each gene. Post-training, GNN explainability methods (GNNExplainer, GraphSVX, SubgraphX) were applied to each confirmed seed gene to identify its most influential network neighbors. Candidate disease genes were ranked by the number of seed-gene subgraphs in which they appeared as highly important neighbors, with ties broken by cumulative importance score. Rankings were evaluated by Precision@k and AUROC against holdout gene sets from DisGeNET and OMIM+PheI databases.
