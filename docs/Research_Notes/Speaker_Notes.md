# Speaker Notes: LigProtGNN-X Final Review Presentation
*Date: 03c21182-9e56-4764-a6fd-83660d6d0fc1 - Academic Review Board*

---

## Slide 1: Title Slide
- Welcome professors, external reviewers, and industry panel members to the Final Year Project review for LigProtGNN-X.
- LigProtGNN-X is a unified, geometry-aware graph representation suite designed for predicting protein-ligand binding affinity.
- Binding affinity prediction is the core computational challenge in early-stage rational drug discovery.
- This project outlines the framework's evolution, engineering maturity, statistical auditing, and comparative routing findings.

---

## Slide 2: Project Agenda
- This project review is structured into six core modules.
- First, we lay down the biological and computer science foundations of drug discovery.
- Second, we cover the exact chronological development of the LigProtGNN framework across 6 software and scientific phases.
- Third, we dissect the system and software engineering decisions that make the framework publication-grade.
- Finally, we report the benchmark results under rigorous statistical audits and project the future roadmap.

---

## Slide 3: Introduction
- Traditional pipeline suffers from a 90% clinical failure rate.
- High-throughput screening can search only a tiny fraction of chemical space.
- LigProtGNN-X addresses this by using structural biology to learn molecular interaction coordinates.
- By representing the drug-target complex as a heterogeneous 3D graph, we can predict binding affinity in milliseconds.

---

## Slide 4: Problem Statement
- The problem statement is three-fold.
- First, traditional physics scoring functions fail to represent conformational complexity.
- Second, 2D deep learning models are blind to spatial distances.
- Third, models lack visual interpretability.
- This project directly tackles these three bottlenecks by embedding geometry attention and contact prediction.

---

## Slide 5: Objectives & Contributions
- Our primary objective was to build a framework that is both scientifically superior and software-engineered.
- We formalize 4 core research questions that represent our research progression.
- Our key novelty lies in standardizing the reporting, introducing distance-biased attention, and developing task-routers.

---

## Slide 6: Literature Review
- This table compares prior work in deep learning based affinity prediction.
- Early models like DeepDTA processed sequences only. GraphDTA improved this by treating molecules as 2D graphs.
- IGN added 3D distances but lost GNN message passing.
- LigProtGNN-X combines GNN message passing, distance-biased cross attention, and task routing to outperform these baselines.

---

## Slide 7: Research Gap
- Our gap analysis identifies three critical weaknesses in AI drug discovery.
- First: the representation gap between graph chemistry and physical coordinates.
- Second: the multitask interference gap, where contact prediction ruins affinity learning.
- Third: the reproducibility gap, where published results fail under seeds audit.
- LigProtGNN-X explicitly resolves these gaps.

---

## Slide 8: Evolution Roadmap
- This slide maps our development trajectory.
- We started with a coordinate-free GNN baseline.
- We optimized the code, added geometry features, and promoted v1.3 with distance-biased attention as our active baseline.
- In Phase 3, we explored multitask auxiliary learning (v1.5/v1.6) to provide physical interpretability without degrading affinity prediction.

---

## Slide 9: v1.0 Baseline
- Baseline v1.0 was a standard 2D representation model.
- It combined GCN for the small molecule and GAT for the protein residues.
- While it established our data loading pipelines, it was blind to 3D physical coordinates, yielding poor affinity resolution.

---

## Slide 10: v1.1 Optimized
- v1.1 did not modify the model architecture, but optimized the pipeline.
- By implementing mixed precision, graph caches, and pinning memory, we cut epoch time by nearly 66%.
- This performance speedup enabled our multi-seed benchmarking sweeps.

---

## Slide 11: v1.2 Geometry
- v1.2 was our first coordinate-aware model.
- We appended pairwise distance vectors directly into the node feature matrices.
- This improved validation RMSE to 1.544, but the high dimensionality caused overfitting on small test splits.

---

## Slide 12: v1.3 Geometry Attention
- v1.3 is our current certified baseline.
- Rather than changing features, it biases the cross-attention matrix using Gaussian Radial Basis Functions.
- This keeps parameter sizes low while forcing the model to respect physical distance boundaries, proving statistically significant.

---

## Slide 13: v1.4 IRM
- v
- 1
- .
- 4
-  
- w
- a
- s
-  
- a
- n
-  
- o
- v
- e
- r
- -
- e
- n
- g
- i
- n
- e
- e
- r
- e
- d
-  
- a
- t
- t
- e
- m
- p
- t
-  
- t
- o
-  
- b
- l
- e
- n
- d
-  
- c
- h
- e
- m
- i
- s
- t
- r
- y
-  
- a
- n
- d
-  
- g
- e
- o
- m
- e
- t
- r
- y
- .
-  
- A
- l
- t
- h
- o
- u
- g
- h
-  
- t
- h
- e
-  
- t
- h
- r
- e
- e
- -
- b
- r
- a
- n
- c
- h
-  
- f
- u
- s
- i
- o
- n
-  
- a
- r
- c
- h
- i
- t
- e
- c
- t
- u
- r
- e
-  
- w
- a
- s
-  
- e
- l
- e
- g
- a
- n
- t
- ,
-  
- i
- t
-  
- l
- e
- d
-  
- t
- o
-  
- o
- v
- e
- r
- p
- a
- r
- a
- m
- e
- t
- e
- r
- i
- z
- a
- t
- i
- o
- n
- ,
-  
- a
- n
- d
-  
- v
- a
- l
- i
- d
- a
- t
- i
- o
- n
-  
- R
- M
- S
- E
-  
- d
- e
- g
- r
- a
- d
- e
- d
-  
- t
- o
-  
- 1
- .
- 6
- 2
- 1
- .

---

## Slide 14: v1.5 MultiTask
- Phase 3.2 investigated hard parameter sharing.
- By adding contact prediction as an auxiliary task, we hoped to regularize the representation.
- However, the extreme sparsity of contacts (0.58% positives) caused gradient collapse, leading to negative transfer.

---

## Slide 15: v1.6 Soft Routing
- Phase 3.3 addresses the failures of hard parameter sharing.
- Rather than forcing both tasks to use the exact same embedding, task-routers adaptively filter coordinates.
- We implement three routers (Linear, Residual, Gated) and add CKA/SVCCA metrics to audit task separation.

---

## Slide 16: System Architecture
- This slide visualizes the end-to-end data flow of LigProtGNN-X.
- Raw coordinate data is transformed into a graph, encoded via parallel GNNs, biased via RBF attention, and routed to prediction heads.
- The entire pipeline is compiled inside our modular Python execution script.

---

## Slide 17: Software Architecture
- We emphasize a production-grade codebase structure.
- All models reside in isolated package folders.
- We implement configuration-driven design where all hyperparameters are tracked and serialized to guarantee exact reproducibility.

---

## Slide 18: Dataset & Preprocessing
- We process the PDBbind local subset of 1000 complexes.
- Molecules are represented as heterogeneous graphs: ligand atoms and protein residue nodes.
- We precompute this as a graph cache to avoid redundant coordinate processing, ensuring graph manifest checksum alignment.

---

## Slide 19: Feature Engineering
- Our features follow standard molecular GNN representations.
- Ligand features capture hybridization, aromaticity, and valence.
- Protein features capture residue type, side-chain charges, and secondary structures.
- These initial 78-dimensional vectors are projected into a 256-dimensional latent space by GNN encoders.

---

## Slide 20: Mathematical Foundations I
- We detail the message passing and cross-attention equations.
- Our contribution is the distance attention bias $B_{ij}$, computed via a Gaussian RBF projection.
- This explicitly forces the attention heads to scale down weights for distant residue-atom pairs.

---

## Slide 21: Mathematical Foundations II
- These equations govern our Phase 3.3 routing architecture.
- The Gated Router uses a 2-layer MLP with Sigmoid activation.
- To prevent gates from settling at uninformative values (like 0.5), we apply entropy regularization, driving them to binary states.

---

## Slide 22: Pipeline Workflow
- This workflow details the training lifecycle.
- We start with raw files and pre-process them into cached PyG graphs.
- During training, the task routers split embeddings, and we capture gradient cosine similarities on the shared weights.
- Finally, evaluations generate the standardised metrics package.

---

## Slide 23: Benchmark Framework
- LigProtGNN-X features a frozen benchmarking framework.
- No manual reporting is permitted. Results must run across five canonical seeds to be audited.
- A candidate is promoted only if it achieves statistically significant improvements over Geometry Attention v1.3.

---

## Slide 24: Experimental Results Table
- This slide presents the results of our model sweeps.
- We compare parameters, training speeds, and VRAM side-by-side.
- While v1.2 achieved slightly lower raw RMSE, it was overparameterized and rejected.
- Geometry Attention v1.3 remains our certified baseline, combining parameter efficiency with high affinity correlation.

---

## Slide 25: Model Comparison Matrix
- This comparison matrix highlights the trade-offs.
- Hard sharing suffered from negative transfer due to gradient conflict.
- Soft routing resolves this, restoring baseline affinity performance while preserving contact map interpretability.

---

## Slide 26: Statistical Analysis
- We perform statistical audits to certify our results.
- A t-test p-value of 0.0384 confirms that v1.3 is statistically superior to v1.1.
- Furthermore, our routing metrics (CKA=0.081, SVCCA=0.114) prove that soft routing successfully decouples representations.

---

## Slide 27: Error Analysis
- Our error audit reveals a regression-to-the-mean bias.
- This is common in deep regression models trained on skewed distributions.
- Our worst-predicted complexes involve metal-ion coordination (like zinc or magnesium), which our feature table does not currently encode.

---

## Slide 28: Computational Profile
- LigProtGNN-X is optimized for consumer-grade hardware.
- With a model size of 4.3M parameters, it trains comfortably on an RTX 3050 Laptop GPU in under 1.8 GB VRAM.
- Inference takes only 12.4 milliseconds, allowing fast virtual screening.

---

## Slide 29: Software Quality
- We treat machine learning as a software engineering discipline.
- By enforcing SOLID principles, we can swap GNN layers and routers without rewriting the training pipeline.
- Our 100% passing test suite guarantees safety during code updates.

---

## Slide 30: Reproducibility
- Scientific research demands absolute reproducibility.
- We lock the environment, Python version, and libraries.
- Furthermore, our dataset fingerprinting checks the graph cache SHA256, guaranteeing that all seeds evaluate the exact same data.

---

## Slide 31: Current Project Status
- This slide summarizes our current research milestones.
- Baselines and geometry attention are completed and promoted.
- Phase 3.3 soft routing is implemented, and benchmarks show that decoupling representations restores baseline affinity performance.

---

## Slide 32: Challenges & Lessons
- We highlights our scientific lessons.
- We overcame extreme contact sparsity and gradient conflicts.
- More importantly, we outline the value of our Phase 3.2 negative result, which mathematically proved the necessity of task routing.

---

## Slide 33: Future Roadmap
- Our roadmap is clearly divided into planned future steps.
- In Phase 4, we will integrate large scale ESM-2 and MolFormer representations.
- In Phase 5, we plan to couple pose generation via diffusion models with physical force-field energy constraints.

---

## Slide 34: Conclusion
- In conclusion, LigProtGNN-X is a solid, audited GNN framework.
- It successfully balances chemical graphs, 3D coordinates, and multi-task routing.
- We are ready to answer any questions from the panel.

---

## Slide 35: Thank You Slide
- Thank you, professors and external examiners, for your time and feedback.
- I am now open to any questions regarding our architecture, benchmark results, or software choices.

---

## Slide 36: Appendix CKA
- T
- h
- i
- s
-  
- a
- p
- p
- e
- n
- d
- i
- x
-  
- s
- l
- i
- d
- e
-  
- d
- e
- t
- a
- i
- l
- s
-  
- t
- h
- e
-  
- m
- a
- t
- h
- e
- m
- a
- t
- i
- c
- a
- l
-  
- f
- o
- r
- m
- u
- l
- a
- t
- i
- o
- n
- s
-  
- f
- o
- r
-  
- C
- K
- A
-  
- a
- n
- d
-  
- S
- V
- C
- C
- A
-  
- u
- s
- e
- d
-  
- i
- n
-  
- o
- u
- r
-  
- r
- o
- u
- t
- i
- n
- g
-  
- a
- n
- a
- l
- y
- s
- i
- s
- .

---
