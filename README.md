# Neural cellular automata for ablation planning

Hands-on 3 of the MICCAI 2026 tutorial *Learning to Self-Organize: Neural
Cellular Automata for Medical Imaging* (Strasbourg, 1 October 2026).

You train a small neural cellular automaton that takes a liver CT slice, a
microwave needle and a power, and predicts which tissue dies after five minutes
of heating. From the same rollout it also segments the liver vessels, which it
is never shown: it has to find them, because they carry heat away. Then you
drag needles around real patient slices while the automaton runs live in your
browser.

The labels come from the physics solvers of my PhD work (Pennes bioheat,
microwave power deposition, Arrhenius damage), developed under the supervision
of Prof. Caroline Essert and Juan Verde.

## Open the notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JonasMht/miccai2026-nca-tutorial/blob/master/ablation2d/notebooks/ablation_with_nca_TUTORIAL.ipynb)

1. Open the link above.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run the first cell. It clones this repository, which carries the data and a
   trained reference model, so there is nothing else to install.

The completed notebook is
[`ablation_with_nca_SOLUTION.ipynb`](https://colab.research.google.com/github/JonasMht/miccai2026-nca-tutorial/blob/master/ablation2d/notebooks/ablation_with_nca_SOLUTION.ipynb),
and `ablation2d/notebooks/ablation_with_nca_SOLUTION.executed.html` is a run of
it with every output, including the live planner, that opens in any browser.

## What is here

- `ablation2d/`: the package, both notebooks, the corpus, the models and the
  tests. Details in [`ablation2d/README.md`](ablation2d/README.md), every
  measured number in [`ablation2d/docs/RESULTS.md`](ablation2d/docs/RESULTS.md).
- `ablation2d/slides/tips_and_tricks_nca.html`: the 10-minute talk that comes
  before the session. Open it in a browser; **P** shows the notes and a timer.

```bash
pip install -r requirements.txt
python -m pytest ablation2d/tests
```

The code is MIT licensed (`LICENSE`). The corpora contain de-identified patient
CT and carry their own terms: see `ablation2d/data/DATASHEET.md`.
