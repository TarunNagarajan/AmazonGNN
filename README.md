# AmazonGNN
 A scalable Graph Neural Network-based recommender system built on the Amazon product dataset. Implements state-of-the-art GNN models like GraphSAGE for personalized product recommendations leveraging graph structure and user-item interactions.

# GNN Recommender Project: 60-Day Intensive Roadmap

**Project Name:** `Nodewise`
**Dataset:** 10% subset of Amazon Product Graph
**Objective:** Build a high-performance GNN-based product recommender system using PyG

| Day | Project Task                                                                | GNN Learning Task                                                               |
| --- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 1   | Set up project repository (`nodewise`) with folders: `data`, `models`, etc. | Read "What is a Graph Neural Network?" (Distill.pub or PyTorch Geometric intro) |
| 2   | Download full Amazon graph dataset                                          | Study types of GNNs: GCN, GraphSAGE, GAT – overview and intuition               |
| 3   | Write a script to extract 10% product graph subset by category sampling     | Deep dive: GraphSAGE paper (Hamilton et al., 2017) – key ideas and architecture |
| 4   | Clean and filter raw dataset (remove duplicates, bad rows, etc.)            | Study GNN node features vs. edge features and graph construction                |
| 5   | Construct NetworkX graph from subset                                        | Hands-on: create toy graphs and visualize message passing in NetworkX           |
| 6   | Convert graph to PyG format: `edge_index`, `x`, and `y` tensors             | Read PyG's documentation on `Data` objects, `InMemoryDataset`                   |
| 7   | Validate and visualize graph (degree distribution, edge density)            | Learn about graph sparsity, connectivity, and their impact on GNNs              |
| 8   | Engineer basic node features (e.g. TF-IDF of product descriptions)          | Read on feature propagation and receptive field in GNNs                         |
| 9   | Split dataset: 80/10/10 train/val/test                                      | Learn common evaluation strategies: link prediction, node classification        |
| 10  | Implement PyG `InMemoryDataset` subclass for loading custom data            | Walk through official PyG tutorial with CORA/Citeseer dataset                   |
| 11  | Baseline 1: Logistic regression on node features                            | Learn limitations of shallow models and motivation for GNNs                     |
| 12  | Baseline 2: MLP model on features only                                      | Study inductive vs transductive learning in GNNs                                |
| 13  | Write modular GNN training loop with PyTorch                                | Learn about training GNNs with mini-batching using `NeighborSampler`            |
| 14  | Implement basic GraphSAGE layer                                             | Study GraphSAGE aggregation types: mean, max, LSTM                              |
| 15  | Train GraphSAGE on sampled data                                             | Learn how neighborhood sampling reduces memory and compute cost                 |
| 16  | Evaluate model: precision\@k, NDCG, recall\@k                               | Understand evaluation metrics in recommender systems                            |
| 17  | Add TensorBoard logging                                                     | Study overfitting and regularization techniques in GNNs                         |
| 18  | Add edge dropout and layer dropout                                          | Study dropout in message passing: impact on GNN generalization                  |
| 19  | Implement validation loop + early stopping                                  | Learn about model selection in graph-based learning                             |
| 20  | Hyperparameter tuning: hidden dim, layers, aggregation                      | Read: "How Powerful are GNNs" (Xu et al. – GIN paper)                           |
| 21  | Add learning rate scheduler                                                 | Understand vanishing gradients and depth vs. expressivity trade-off             |
| 22  | Add support for edge features (if available)                                | Learn about edge-conditioned GNNs (e.g. EdgeConv)                               |
| 23  | Add multi-category node embeddings                                          | Read about heterogeneous graphs and multiplex networks                          |
| 24  | Train model on larger subset (e.g., 25%) for benchmarking                   | Learn batch training vs. full-batch in GNNs                                     |
| 25  | Profile training time and GPU memory usage                                  | Study scalability and graph partitioning strategies                             |
| 26  | Write a `config.yaml` and argparse interface                                | Learn reproducibility best practices for graph experiments                      |
| 27  | Begin integrating attention (GAT) layer                                     | Study: Graph Attention Networks (Velickovic et al., 2018)                       |
| 28  | Compare GAT vs. GraphSAGE in performance                                    | Understand attention head aggregation and softmax normalization                 |
| 29  | Optimize data loading pipeline for PyG                                      | Learn `NeighborLoader`, `ClusterLoader`, and efficient graph sampling in PyG    |
| 30  | Mid-project checkpoint: write a blog/README update                          | Review: GNN taxonomy – spectral vs spatial, inductive vs transductive           |
| 31  | Implement negative sampling for link prediction                             | Learn about contrastive learning in graphs (e.g., DGI, GraphCL)                 |
| 32  | Add edge prediction head to the model                                       | Study bipartite graphs and user-item edge modeling                              |
| 33  | Train edge-prediction GNN variant                                           | Learn how to train GNNs for recommendation (vs classification)                  |
| 34  | Evaluate with hit\@k, MRR, MAP                                              | Study evaluation of top-k recommendation systems                                |
| 35  | Build a nearest-neighbors index for item recommendations                    | Understand approximate nearest neighbor (ANN) search strategies                 |
| 36  | Add online inference script (`recommend(user_id)`)                          | Study fast GNN inference strategies (caching, quantization)                     |
| 37  | Implement embedding cache to speed up inference                             | Study embedding reuse and cold-start problem                                    |
| 38  | Profile latency and throughput on CPU and GPU                               | Study GNN inference bottlenecks                                                 |
| 39  | Export embeddings to file (`.npy`, `.pt`)                                   | Learn about downstream use of graph embeddings                                  |
| 40  | Write tests for data loading + core GNN layers                              | Learn testing best practices in ML pipelines                                    |
| 41  | Refactor GNN model into modular `models/` folder                            | Read clean code tips for ML codebases                                           |
| 42  | Write CLI script to train/eval with config                                  | Learn scriptable ML pipelines and CLI tools                                     |
| 43  | Prepare graphs for visualization in TensorBoard or DGL                      | Study graph visualization tools                                                 |
| 44  | Add metadata enrichment (price, rating) if available                        | Understand graph enrichment with side-information                               |
| 45  | Train hybrid model (features + structure + metadata)                        | Learn feature fusion in graph models                                            |
| 46  | Add node2vec baseline for comparison                                        | Study node2vec and shallow embedding techniques                                 |
| 47  | Train + compare all models on same split                                    | Learn benchmarking GNNs                                                         |
| 48  | Write `eval.py` for standardized benchmarking                               | Understand experiment tracking and ablation studies                             |
| 49  | Add experiment tracker (Weights & Biases or CSV)                            | Learn experiment versioning and tracking                                        |
| 50  | Start writing final report + README                                         | Review entire project from problem to architecture to results                   |
| 51  | Design quantitative results table + graphs                                  | Learn result communication and visualization                                    |
| 52  | Write architecture diagram                                                  | Learn best practices for technical documentation                                |
| 53  | Refactor final repo layout                                                  | Study open-source best practices for ML repos                                   |
| 54  | Polish README with install, run, usage instructions                         | Study how to write reproducible research repos                                  |
| 55  | Add license, citation, and references                                       | Learn open licensing and research citation norms                                |
| 56  | Write final model card                                                      | Study model card documentation standards                                        |
| 57  | Create demo notebook for showcasing recommendations                         | Learn interactive model demos for GNNs                                          |
| 58  | Push final version to GitHub                                                | Study GitHub project showcase strategies                                        |
| 59  | Share project on LinkedIn + GitHub README badge                             | Learn personal branding for ML projects                                         |
| 60  | Submit to Papers With Code or similar repo                                  | Learn about public ML leaderboards and academic discoverability                 |
