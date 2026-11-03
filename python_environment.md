# Python environment migration

## Supported environment

The default course environment uses Python 3.11, managed by uv. The root
[`pyproject.toml`](pyproject.toml) is the dependency source of truth and
[`uv.lock`](uv.lock) records the resolved packages. The version is selected
to retain the open-source H2O-3 Python client: its current published
compatibility page lists Python 3.7–3.11 and requires NumPy below 2. Current
SHAP releases that work with the recent XGBoost base-score format require
NumPy 2, so H2O-3 runs from a mutually exclusive, NumPy-1.x environment.
H2O-3 Secure documentation describes a newer Python range, but its install
page requires contacting H2O for the package; this course setup does not
assume access to that distribution.

The complete default stack and H2O environment were exercised on macOS
Apple Silicon. Linux and Windows setup notes below cover the documented CPU
runtime prerequisites, but were not verified as complete course environments
in this migration.

## Setup

Install uv, then from the repository root:

```sh
uv python install 3.11
uv sync
uv run jupyter lab
```

For local XGBoost on macOS, install its OpenMP runtime once with
`brew install libomp`. Linux CPU installations use the system OpenMP runtime;
Windows users need the Visual C++ Redistributable. XGBoost's GPU wheel is
available on supported Linux platforms; macOS wheels are CPU-only.

H2O-backed examples use a separate environment so NumPy 1.x does not replace
the default NumPy 2 environment:

```sh
UV_PROJECT_ENVIRONMENT=.venv-h2o uv sync --group legacy-h2o --no-default-groups
```

H2O-3 also requires a 64-bit Java runtime. Its documentation lists Java 8–17.
For the validated H2O-3 3.46 setup on Apple Silicon, use an ARM64 Java 8
runtime and set `JAVA_HOME` to its `Contents/Home` directory when launching an
H2O example. Do not sync `course` and `legacy-h2o` together: uv records this as
an explicit group conflict.

For neural-network experiments, TensorFlow is an optional extra:

```sh
uv sync --extra neural
```

## Compatibility and limits

- The default locked environment was resolved and synchronized on macOS
  Apple Silicon with Python 3.11.17. Core imports and small EBM, XGBoost, and
  SHAP runs passed with Interpret 0.7.8, XGBoost 3.2.0, SHAP 0.51.0, NumPy
  2.4.6, and pandas 2.3.3.
- The H2O-3 group resolved and imports passed with H2O 3.46.0.12, Interpret
  0.7.8, SHAP 0.49.1, XGBoost 2.1.4, and NumPy 1.26.4. An actual H2O server
  launch and GBM fit passed on Apple Silicon with Python 3.11.17 and Azul Zulu
  Java 8 ARM64 (8.0.512). The smoke model used 60 generated rows and
  `min_rows=1`. The migrated `util.evaluate.cv_model_rank` also ran against a
  live H2O GBM and returned the expected 20 fold/metric rows. A Java 17 attempt
  failed in H2O's native XGBoost library loader; although Java 17 is in H2O's
  published range, it is not the validated runtime for this client/host
  combination.
- PiML is not part of the default environment. Its vendor installation guide
  documents Python only through 3.10 and pins old NumPy, pandas,
  scikit-learn, and XGBoost ranges. The course's PiML example notebook and
  requirements note are archived in `deprecated/assignment/assignment_1/`;
  they are not supported by this setup.
- Notebook files linked from README or assignment handouts remain unchanged.
- Supplement pages `am1.md` through `am5.md` and `am7.md` remain in `additional_material/`.
  Superseded Python setup files are archived in
  `deprecated/additional_material/`; previous Lecture 7 resources are in
  `deprecated/lecture_7/` (the former language-model risk-mitigation lecture).

## Verification performed

- `uv lock --python 3.11` resolved the course and H2O-compatible group forks;
  `uv sync` and a separate `legacy-h2o` sync completed.
- Core stack smoke run: EBM fit/predict, XGBoost fit, and SHAP TreeExplainer
  returned values on generated binary-classification data.
- H2O-3 3.46.0.12 launched its server under Azul Zulu Java 8 ARM64, fit a GBM
  on generated data, and produced AUC 1.0. The migrated
  `util.evaluate.cv_model_rank` also ran against a live H2O GBM. The tested
  Java archive lived in `/private/tmp`; Java was not installed system-wide.
- The linked Assignment 5 template was run against the repository's HMDA
  training data in a temporary copy. Its full EBM fit exceeded a four-minute
  per-cell timeout. This is an incomplete full-notebook run, not a
  compatibility pass; the linked notebook source was left unchanged.
- The remaining linked lecture and assignment notebooks that start H2O or
  perform large course analyses were not live-run in this pass. Their sources
  remain intact.

## Change inventory and deferred examples

The migration added the root uv manifest/lock and Python pin, clarified the
README's local-versus-Colab setup, modernized pandas handling in
`util/evaluate.py` and `util/explain.py`, and updated the assignment
requirement pointers.

These existing files remain at their original paths:

- Lecture notebooks, assignment templates, and the evaluation notebook remain
  at their linked paths. A full run of the Assignment 5 template exceeded a
  four-minute per-cell timeout during its EBM fit; it was not rewritten.
- Supplement pages `am1.md` through `am5.md` and `am7.md` remain in `additional_material/`.
  Superseded Python setup files are archived in
  `deprecated/additional_material/`; previous Lecture 7 resources are in
  `deprecated/lecture_7/` (the former language-model risk-mitigation lecture).
- PiML remains outside the uv environment. Its public installation guide
  documents Python 3.7–3.10 and older NumPy/pandas/scikit-learn constraints.
  The course example and requirements note are archived in
  `deprecated/assignment/assignment_1/`; no current supported combination
  was verified, and no end-of-support claim is made.
- The main supported environment is CPU-first. XGBoost's platform matrix
  documents GPU wheels for Linux and Windows, while macOS is CPU-only; no GPU
  setup or acceleration benchmark is part of this migration.

## Compatibility sources

- [H2O-3 Python and Java requirements](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html)
- [H2O-3 Secure requirements](https://docs.h2o.ai/h2o-3-secure/welcome) and [package access](https://docs.h2o.ai/h2o-3-secure/downloading)
- [SHAP release notes](https://shap.readthedocs.io/en/stable/release_notes.html)
- [InterpretML installation guide](https://interpret.ml/docs/installation-guide.html)
- [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html)
- [PiML installation guide](https://selfexplainml.github.io/PiML-Toolbox/_build/html/install.html)
