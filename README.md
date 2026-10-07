# CB-RBM

Reference implementation of the Context-Boosted Restricted Boltzmann Machine
from

> Firas Bayram and Maxime Cordy. Feature-Context Consistency: Unsupervised
> Adversarial Detection with Drift Stability on Attributed Graphs. NeurIPS 2026.

A Gaussian-Bernoulli RBM is trained on the nodes of an attributed graph that
are assumed legitimate. Its hidden units receive a gated context term built
from the sign-corrected mean of each node's neighbor features. The Context
Boost score, the free-energy reduction due to context, is high for nodes whose
features agree with their neighborhood and low for injected nodes that lack
such context. The detection threshold is the 95th percentile of the negative
score on a held-out slice of the same pool, so no anomaly labels are used.

## Install

```
pip install -e .
```

Requires Python 3.9+, PyTorch, NumPy and scikit-learn.

## Usage

```python
import numpy as np
from cbrbm import CBRBMDetector

det = CBRBMDetector(n_hidden=256, epochs=100)
det.fit(x, edge_index, pool_idx)        # x: (n, d) features, edge_index: (2, E), pool_idx: nodes assumed legitimate

scores = det.score_graph(x, edge_index) # s_i = -CB(x_i, c_i)
flags = det.predict(scores)             # s_i >= threshold

det.refresh(x_new, edge_index_new, window_idx)   # re-estimate sign vector and threshold, weights fixed
```

`cbrbm.attacks` implements the four injected-adversary attacks of the paper
(isolated injection, feature camouflage, Sybil, relation camouflage) and
`cbrbm.drift` the random-walk drift with context recomputation.

`examples/synthetic.py` runs the full pipeline on a synthetic homophilic graph:

```
python examples/synthetic.py
```

## Data

The experiments in the paper use the XBlock Ethereum phishing graph and the
Reddit banned-user graph, both public. Feature construction is described in
the paper's appendix. This repository contains the method, not the dataset
pipelines or the experiment scripts.

## Citation

```bibtex
@inproceedings{bayram2026cbrbm,
  title     = {Feature-Context Consistency: Unsupervised Adversarial Detection with Drift Stability on Attributed Graphs},
  author    = {Bayram, Firas and Cordy, Maxime},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT
