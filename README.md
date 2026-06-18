# RAHGRec

**RAHGRec** (**R**ole-**A**ware **H**yper**G**raph **Rec**ommendation) is a role-aware hypergraph learning framework for **third-party library (TPL) recommendation** in the Python software ecosystem. It combines project-library adoption behavior with typed semantic information from a software knowledge graph.

**Paper title:**  
*RAHGRec: A Role-Aware Hypergraph-Based Framework for Project-Library Recommendation in Python Ecosystem with Knowledge-Graph Relations*

---

## Replication Package Overview

This replication package is intended to support reproduction of the experiments reported in the paper. It contains the implementation of RAHGRec, processed data files, knowledge graph construction inputs, experiment configurations, and scripts for training and evaluation.

The package is organized to support the following tasks:

- preparing the project-library interaction data,
- constructing the software knowledge graph,
- training RAHGRec,
- running baseline comparisons,
- reproducing recommendation metrics,
- reproducing sensitivity, scalability, and ablation experiments.

---

## Environment Requirements

The framework has been tested under **Python 3.7.5**. Newer Python versions may also work, but the versions below correspond to the tested environment.

| Dependency | Version tested |
|---|---:|
| CUDA | 10.2 |
| PyTorch | 1.11.0 |
| NumPy | 1.21.5 |
| Pandas | 1.3.5 |
| SciPy | 1.4.1 |
| tqdm | 4.64.0 |
| scikit-learn | 0.22 |

Install the tested dependencies with:

```bash
pip install torch==1.11.0 numpy==1.21.5 pandas==1.3.5 scipy==1.4.1 tqdm==4.64.0 scikit-learn==0.22
```

If CUDA is not available, the code can be run on CPU for small-scale checks, but GPU execution is recommended for reproducing the full experiments.

---

## Hardware Used in the Paper

The experiments reported in the paper were run on the following setup:

- CPU: Intel Core i7
- Memory: 32 GB DDR4
- GPU: NVIDIA RTX A5000
- CUDA: 10.2
- Python: 3.7.5
- PyTorch: 1.11.0

Runtime may vary depending on the GPU, CPU, memory, and software environment.

---

## Data Description

The experiments use the PyRec benchmark for Python project-library recommendation. The processed dataset contains:

- 12,421 Python projects,
- 963 distinct libraries,
- 121,474 project-library interactions,
- 73,277 semantic relationships,
- 9,675 auxiliary knowledge graph entities.

The interaction data are derived from import relationships in GitHub repositories and associated PyPI metadata. Semantic information includes project tags, library dependencies, authors, licenses, topics, keywords, and descriptive metadata where available.

Projects associated with fewer than five libraries are filtered out before evaluation, following the experimental protocol described in the paper.

---

## Knowledge Graph Construction

The software knowledge graph is constructed from the processed project-library dataset and associated metadata. The KG is not assumed to be a manually curated external resource.

The graph contains the following node types:

- projects,
- libraries,
- maintainers or authors,
- licenses,
- topics,
- keywords,
- descriptive tags and metadata entities.

The graph contains the following relation types:

- project-library usage relations,
- library dependency relations,
- maintainer or author relations,
- license relations,
- topic, keyword, and descriptive metadata relations,
- project-project similarity relations,
- library-library similarity relations.

Similarity-based relations are constructed only between comparable entity types. Project-project similarity is computed from topic-set overlap, while library-library similarity is computed from metadata-set overlap. Jaccard similarity thresholds are used to retain meaningful similarity edges.

Inverse triples are added during preprocessing to support bidirectional information flow in the model.

---

## Evaluation Protocol

The experiments use a masked implicit-feedback evaluation protocol.

For each project:

1. A subset of its adopted libraries is masked.
2. The remaining interactions are used as observed training evidence.
3. The masked libraries are treated as the ground-truth relevant libraries.
4. The model ranks candidate libraries not retained in the project training interactions.
5. The top-K ranked libraries are evaluated against the masked ground truth.

The removal ratio is varied as:

```text
r_m = {20%, 40%, 60%}
```

Recommendation list sizes are evaluated at:

```text
K = {5, 10, 20}
```

Each experimental setting is repeated 50 times with different random splits, and the mean result is reported.

---

## Evaluation Metrics

The package reports the following metrics:

- Mean Precision (MP)
- Mean Recall (MR)
- Mean F-score (MF)
- Mean Reciprocal Rank (MRR)
- Coverage (COV)

Precision measures the fraction of recommended libraries that are relevant. Recall measures the fraction of withheld libraries that are recovered. F-score is the harmonic mean of precision and recall. MRR evaluates how early the first relevant library appears in the ranked list. Coverage measures the fraction of distinct libraries that appear in recommendations across all query projects.

---

## Running RAHGRec

After installing the dependencies and preparing the data, run the main training script with the configuration used in the paper.

Example:

```bash
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --topk 5 10 20
```

If your local repository uses a different script name, use the equivalent entry point provided in the implementation. The important settings to match the paper are:

- model: `RAHGRec`
- dataset: `pyrec`
- removal ratios: `0.2`, `0.4`, `0.6`
- top-K values: `5`, `10`, `20`
- embedding dimension: `128`
- propagation layers: `2`

---

## Reproducing Table 3

To reproduce the main recommendation-quality comparison:

1. Run RAHGRec under each removal ratio.
2. Run all baseline methods under the same splits and metric definitions.
3. Evaluate MP, MR, MF, MRR, and COV at K = 5, 10, and 20.
4. Average results across the repeated random splits.

Example:

```bash
python main.py --model RAHGRec --dataset pyrec --rm 0.2 --topk 5 10 20
python main.py --model RAHGRec --dataset pyrec --rm 0.4 --topk 5 10 20
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --topk 5 10 20
```

The same evaluation protocol should be used for all baselines.

---

## Reproducing Sensitivity Experiments

The paper reports sensitivity analyses for propagation depth, semantic thresholds, and embedding dimensionality.

Typical settings include:

- propagation layers: 1 to 5,
- embedding dimensions: 32, 64, 128, 256,
- project and library similarity thresholds as reported in the paper.

Example:

```bash
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --layers 1
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --layers 2
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --layers 3
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --layers 4
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --layers 5
```

---

## Reproducing Scalability Experiments

The scalability experiment evaluates RAHGRec under increasing dataset sizes while keeping the removal ratio fixed at 60%.

Dataset sizes used in the paper:

```text
100, 500, 1000, 2000, 4000 projects
```

Example:

```bash
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --sample_size 100
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --sample_size 500
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --sample_size 1000
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --sample_size 2000
python main.py --model RAHGRec --dataset pyrec --rm 0.6 --sample_size 4000
```

---

## Reproducing Ablation Experiments

The ablation study evaluates the contribution of the major RAHGRec components, including:

- removal of the role-aware hypergraph component,
- removal of semantic propagation,
- removal of role gating,
- removal of KG relational learning where applicable.

Example:

```bash
python main.py --model RAHGRec --dataset pyrec --ablation no_hypergraph
python main.py --model RAHGRec --dataset pyrec --ablation no_semantic
python main.py --model RAHGRec --dataset pyrec --ablation no_role_gate
```

---

## Expected Outputs

Each experiment should produce:

- metric results for MP, MR, MF, MRR, and COV,
- training time,
- memory usage where measured,
- log files for each run,
- saved result tables or CSV files.

The reported values may show small numerical differences across systems due to random seeds, GPU behavior, and library versions.

---

## Notes on Reproducibility

To reproduce the reported experiments as closely as possible:

- use the tested dependency versions,
- use the same dataset splits where provided,
- keep the same removal ratios and top-K values,
- run all methods under the same hardware and software environment when comparing runtime,
- repeat each setting 50 times when reproducing the averaged results.

---

## Citation

If you use this package, please cite the associated paper:

```bibtex
@article{rahgrec2026,
  title={RAHGRec: A Role-Aware Hypergraph-Based Framework for Project-Library Recommendation in Python Ecosystem with Knowledge-Graph Relations},
  author={Jamwal, Abhinav and Kumar, Sandeep},
  journal={ACM Transactions},
  year={2026}
}
```

---

## Contact

For questions about the replication package, please contact:

```text
Abhinav Jamwal
Indian Institute of Technology Roorkee
abhinav_j@cs.iitr.ac.in
```
