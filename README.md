# RAHGRec

**RAHGRec** (**R**ole-**A**ware **H**yper**G**raph **Rec**ommendation) is a role-aware hypergraph learning framework for third-party library recommendation in the Python software ecosystem. It combines project-library adoption behavior with typed semantic information from a software knowledge graph.

**Paper title:**  
*RAHGRec: A Role-Aware Hypergraph-Based Framework for Project-Library Recommendation in Python Ecosystem with Knowledge-Graph Relations*

---

## Repository Structure

```text
RAHGRec/
├── datasets/
│   ├── Original/
│   ├── PyLib/
│   │   ├── t01/
│   │   ├── t02/
│   │   ├── t04/
│   │   ├── t06/
│   │   ├── Readme.txt
│   │   └── readme.md
│   ├── result/
├── RAHGREC.py
├── main.py
├── loader_base.py
├── loader_rahgrec.py
├── parser_rahgrec.py
├── model_helper.py
├── log_helper.py
└── README.md
```

The main processed dataset used for the experiments is provided in:

```text
datasets/PyLib/
```

---

## Environment Requirements

The framework has been tested under **Python 3.7.5**.

| Dependency | Version tested |
|---|---:|
| CUDA | 10.2 |
| PyTorch | 1.11.0 |
| NumPy | 1.21.5 |
| Pandas | 1.3.5 |
| SciPy | 1.4.1 |
| tqdm | 4.64.0 |
| scikit-learn | 0.22 |

Install dependencies with:

```bash
pip install torch==1.11.0 numpy==1.21.5 pandas==1.3.5 scipy==1.4.1 tqdm==4.64.0 scikit-learn==0.22
```

GPU execution is recommended for reproducing the full experiments.

---

## Dataset

The dataset is stored in:

```text
datasets/PyLib/
```

It contains processed project-library interaction files and knowledge graph related files used by RAHGRec. The subfolders correspond to different experimental settings used in the paper.

The dataset includes:

- Python software projects,
- third-party libraries,
- project-library usage interactions,
- semantic relationships used to construct the knowledge graph,
- auxiliary metadata entities.

Projects with fewer than five associated libraries are filtered out before evaluation, following the protocol described in the paper.

---

## Knowledge Graph Construction

RAHGRec constructs a software knowledge graph from the processed project-library data and metadata. The graph includes projects, libraries, and auxiliary metadata entities such as authors, dependencies, licenses, topics, keywords, and descriptive tags.

The main relation types include:

- project-library usage relations,
- library dependency relations,
- maintainer or author relations,
- license relations,
- topic and keyword relations,
- project-project similarity relations,
- library-library similarity relations.

Similarity-based relations are constructed only between comparable entity types. Project-project similarity is computed using topic overlap, while library-library similarity is computed using metadata overlap. Inverse triples are added to support bidirectional information flow during propagation.

---

## Evaluation Protocol

The experiments use a masked implicit-feedback protocol.

For each project:

1. A subset of adopted libraries is masked.
2. The remaining libraries are used as training evidence.
3. The masked libraries are treated as ground truth.
4. The model ranks candidate libraries not retained in the project training interactions.
5. The top-K libraries are evaluated against the masked ground truth.

The removal ratios used in the paper are:

```text
20%, 40%, 60%
```

The recommendation cutoffs are:

```text
K = 5, 10, 20
```

Each setting is repeated with multiple random splits, and the mean performance is reported.

---

## Evaluation Metrics

The reported metrics are:

- Mean Precision (MP)
- Mean Recall (MR)
- Mean F-score (MF)
- Mean Reciprocal Rank (MRR)
- Coverage (COV)

These metrics evaluate recommendation accuracy, ranking quality, and diversity.

---

## Running RAHGRec

The main entry point is:

```text
main.py
```

A typical run can be executed as:

```bash
python main.py
```

If command-line arguments are used in your local setup, check:

```text
parser_rahgrec.py
```

for available options such as dataset path, embedding dimension, propagation layers, removal ratio, and top-K values.

The main implementation files are:

```text
RAHGREC.py
loader_rahgrec.py
parser_rahgrec.py
main.py
```

---

## Reproducing Main Results

To reproduce the main recommendation results:

1. Use the processed dataset in `datasets/PyLib/`.
2. Run RAHGRec using `main.py`.
3. Evaluate the model under the removal ratios used in the paper.
4. Report MP, MR, MF, MRR, and COV for `K = 5, 10, 20`.
5. Repeat each setting using the same split protocol described in the paper.

The output recommendation files and result files are stored under:

```text
datasets/result/
```

and the recommendation CSV files in the `datasets/` directory.

---

## Reproducing Additional Experiments

The paper also reports experiments for:

- propagation depth,
- embedding dimensionality,
- similarity thresholds,
- scalability under different dataset sizes,
- ablation settings.

These can be reproduced by changing the corresponding settings in the parser or configuration options and rerunning `main.py`.

---

## Expected Outputs

A completed run should produce:

- recommendation lists,
- MP, MR, MF, MRR, and COV scores,
- log files,
- result CSV files,
- training time and memory usage where measured.

Small numerical differences may occur due to random seeds, hardware, CUDA behavior, and library versions.

---

## Reproducibility Notes

To reproduce the reported results as closely as possible:

- use the tested dependency versions,
- use the dataset in `datasets/PyLib/`,
- keep the same removal ratios and top-K values,
- run all methods under the same evaluation protocol,
- repeat experiments using the same split strategy.

---

## Citation

If you use this repository, please cite the associated paper:

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

```text
Abhinav Jamwal
Indian Institute of Technology Roorkee
abhinav_j@cs.iitr.ac.in
```
