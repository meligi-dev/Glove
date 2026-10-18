# GloVe: Global Vectors for Word Representation

This directory contains implementations and explorations of **GloVe (Global Vectors for Word Representation)**, introduced by Pennington et al. (Stanford, 2014). GloVe is an unsupervised learning algorithm that combines the advantages of two major model families: **global matrix factorization** (such as LSA) and **local context window methods** (such as Word2Vec Skip-gram).

The repository covers both the foundational mechanics from scratch and practical applications using pre-trained embeddings:
1. **[GloVE.ipynb](file:///d:/Learn_NLP/GloVE/GloVE.ipynb)**: From-scratch implementation of the core components of GloVe in Python and PyTorch.
2. **[GloVE(Gensim).ipynb](file:///d:/Learn_NLP/GloVE/GloVE(Gensim).ipynb)**: Loading, analyzing, evaluating analogies, and visualizing pre-trained Stanford GloVe embeddings using Gensim and PCA.

---

## 🧠 Theoretical Foundations

### 1. Motivation: Combining Matrix Factorization & Local Windows
* **Matrix Factorization (e.g., LSA)**: Leverages global statistical co-occurrence information across the entire corpus, but performs poorly on word analogy tasks and weights all co-occurrences equally.
* **Skip-Gram / CBOW (Word2Vec)**: Captures complex linguistic patterns and semantic analogies through local sliding windows, but fails to directly utilize global co-occurrence statistics.
* **GloVe**: Directly models the global word-word co-occurrence matrix $X$, using log-bilinear dot products to encode the ratio of co-occurrence probabilities.

### 2. The Ratio of Co-occurrence Probabilities
Let $P_{ik} = P(k \mid i) = \frac{X_{ik}}{X_i}$ be the probability that word $k$ appears in the context of word $i$. GloVe demonstrates that semantic relevance is captured by ratios of probabilities rather than raw probabilities:

$$\frac{P(k \mid \text{ice})}{P(k \mid \text{steam})}$$

* For $k = \text{solid}$, the ratio is large ($\gg 1$).
* For $k = \text{gas}$, the ratio is small ($\ll 1$).
* For $k = \text{water}$ or $k = \text{fashion}$, the ratio is close to $1$.

### 3. Objective Loss Function
GloVe formulates a weighted least-squares regression model:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

Where:
* $w_i \in \mathbb{R}^d$: Word vector for center word $i$.
* $\tilde{w}_j \in \mathbb{R}^d$: Context word vector for context word $j$.
* $b_i, \tilde{b}_j$: Scalar biases for word $i$ and context word $j$.
* $X_{ij}$: Number of times word $j$ appears in the context of word $i$.
* $f(X_{ij})$: Non-linear weighting function.

### 4. Weighting Function $f(X_{ij})$
To prevent very frequent co-occurrences (e.g., stopwords like *"the"*, *"is"*) from dominating the objective and to avoid undefined $\log(0)$, GloVe introduces a continuous cutoff weighting function:

$$f(x) = \begin{cases} \left(\frac{x}{x_{\max}}\right)^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

Standard hyperparameters proposed by the authors: $x_{\max} = 100$ and $\alpha = 0.75$.

```
   f(x) ^
    1.0 |                +------------------------- (f(x) = 1 for x >= x_max)
        |               /
        |             /  
        |           /    f(x) = (x / x_max)^0.75
        |         /
        |       /
        |    /
      0 +---+--------------------------------------> x
        0   10        x_max (100)
```

---

## 🔬 Notebook Highlights

### 1. `GloVE.ipynb`: Building the Pipeline From Scratch
* **Corpus Tokenization & Vocabulary Mapping**:
  * Tokenizes documents, flattens vocabulary, and handles out-of-vocabulary terms with `<UNK>`.
  * Generates bidirectional lookups (`word2indx` and `indx2word`).
* **Context Window Extraction**:
  * Scans sentences using a sliding window to collect `(center, context)` co-occurrence pairs.
  * Provides a `Random_batch` sampler for mini-batch stochastic training.
* **Co-occurrence Matrix Construction**:
  * Utilizes `collections.Counter` to tabulate co-occurrence counts $X_{ij}$.
* **Weighting Function**:
  * Implements $f(X_{ij}) = (X_{ij} / X_{\max})^\alpha$ with edge-case handling for zero and low counts.

---

### 2. `GloVE(Gensim).ipynb`: Exploring Pre-trained Embeddings
Utilizes the **GloVe 6B** dataset (trained on Wikipedia 2014 + Gigaword 5, 400,000-word vocabulary, 100-dimensional vectors) loaded via Gensim's `KeyedVectors`:

#### A. Semantic Similarity & Nearest Neighbors
Identifies nearest conceptual neighbors via cosine similarity:
* **Political entities** (`obama`):
  * `barack` (0.937), `bush` (0.927), `clinton` (0.896), `mccain` (0.888)
* **Animals** (`cat`):
  * `dog` (0.880), `rabbit` (0.742), `cats` (0.732), `monkey` (0.729)

#### B. Vector Arithmetic & Analogies
Solves classic linear substructure analogies:

$$\vec{w}_{\text{woman}} + \vec{w}_{\text{king}} - \vec{w}_{\text{man}} \approx \vec{w}_{\text{queen}}$$

* **Result**: `queen` with a cosine similarity score of **0.770**.

#### C. Semantic Distance Comparison
Quantifies conceptual proximity using cosine distance ($1 - \text{cosine\_similarity}$):
* $\text{distance}(\text{dog}, \text{cat}) = \mathbf{0.120}$ *(closely related)*
* $\text{distance}(\text{dog}, \text{fruit}) = \mathbf{0.623}$ *(semantically distant)*

#### D. 2D Dimensionality Reduction & Visualization
Reduces 100-dimensional embeddings to 2D using **Principal Component Analysis (PCA)** to visualize semantic clusters across distinct conceptual categories:
* ☕ **Beverages**: *coffee, tea, beer, wine, brandy, rum, champagne, water*
* 🍕 **Food**: *spaghetti, borscht, hamburger, pizza, falafel, sushi, meatballs*
* 🐾 **Animals**: *dog, horse, cat, monkey, parrot, koala, lizard, frog, wolf*
* 🌍 **Countries**: *france, germany, hungary, luxembourg, australia, fiji, china*
* 🎓 **Education**: *homework, assignment, exam, test, class, school, college, university*

The resulting 2D projection demonstrates clear separation between non-overlapping categories while clustering related concepts tightly together.

---

## 📊 Summary of Experimental Results

| Task / Query | Target / Output | Metric / Score | Insight |
| :--- | :--- | :---: | :--- |
| `most_similar("obama")` | `barack`, `bush`, `clinton` | $0.896 - 0.937$ | Captures political and presidential context |
| `most_similar("cat")` | `dog`, `rabbit`, `monkey` | $0.729 - 0.880$ | Clusters domestic and mammalian animals |
| `woman + king - man` | `queen` | $0.770$ | Demonstrates gender-role vector arithmetic |
| `distance("dog", "cat")` | Semantic proximity | $0.120$ | High correlation between domestic pets |
| `distance("dog", "fruit")` | Semantic separation | $0.623$ | Disparate domains produce large distance |

---

## 📁 Repository Structure

```
GloVE/
├── GloVE.ipynb             # From-scratch implementation of GloVe components in PyTorch/Python
├── GloVE(Gensim).ipynb     # Pre-trained vector loading, analogies, distance, and PCA clustering
├── README.md               # Detailed project documentation and theoretical background
└── glove.6B/               # Pre-trained Stanford GloVe vectors
    ├── glove.6B.50d.txt    # 50-dimensional vectors
    ├── glove.6B.100d.txt   # 100-dimensional vectors (used in notebooks)
    ├── glove.6B.200d.txt   # 200-dimensional vectors
    └── glove.6B.300d.txt   # 300-dimensional vectors
```

---

## 🚀 Setup and Usage

### 1. Prerequisites
Install required Python libraries:
```bash
pip install numpy torch matplotlib gensim scikit-learn
```

### 2. Pre-trained Embeddings
The notebook uses `glove.6B.100d.txt`. If downloading GloVe anew, obtain the vectors from Stanford NLP:
```bash
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d GloVE/glove.6B/
```

### 3. Running the Notebooks
Launch Jupyter Notebook or JupyterLab:
```bash
# To run the from-scratch pipeline
jupyter notebook GloVE/GloVE.ipynb

# To run pre-trained embeddings analysis and PCA visualization
jupyter notebook GloVE/GloVE\(Gensim\).ipynb
```

---

## 📚 References
* Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1532–1543. [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
