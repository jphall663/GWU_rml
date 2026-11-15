### Lecture 6 Additional Software Tools

* **Python — Tabular Foundation Models**:
  * [NVIDIA Structured Data Models](https://github.com/NVIDIA/structured-data-models) — Kumo Tabular and other pretrained models for structured data; used in the lecture example.
  * [TabPFN](https://github.com/PriorLabs/TabPFN) — pretrained tabular classification and regression with a scikit-learn-style interface.
  * [TabICL](https://github.com/soda-inria/tabicl) — tabular in-context learning, with inference and pretraining code.
  * [TabPFN Extensions](https://github.com/PriorLabs/tabpfn-extensions) — additional tools for explanations, feature selection, embeddings, and unsupervised applications.

* **Python — Additive Models**:
  * [NAMpy](https://github.com/AnFreTh/NAMpy) — neural additive models with feature-specific neural networks; used for the lecture's comparison model.

### Lecture 6 Additional Software Examples

* [Lecture 6: Neural Additive Models and Tabular Foundation Models](../lecture_6.ipynb) — frozen single-feature Kumo predictions combined by penalized logistic regression, compared with NAMpy; includes data partitioning, shape plots, and local contributions. See the [course environment instructions](../README.md#foundation-model-neural-notebook) for setup.
* [NVIDIA Structured Data Models examples](https://github.com/NVIDIA/structured-data-models/tree/main/examples) — official structured-data modeling examples.
* [TabPFN demonstration notebook](https://github.com/PriorLabs/TabPFN/blob/main/examples/notebooks/TabPFN_Demo_Local.ipynb) — installation, classification, and regression.
* [TabICL basic usage](https://github.com/soda-inria/tabicl#basic-usage) — classification, regression, and cached inference.
* [NAMpy examples](https://github.com/AnFreTh/NAMpy/tree/main/examples) — additive neural modeling examples.

### Lecture 6 Recommended Additional Reading

* **Attention and Prior-data Fitted Networks**:
  * Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
  * Müller et al. (2022), [*Transformers Can Do Bayesian Inference*](https://openreview.net/forum?id=KSugKcbNf9).

* **Tabular Foundation Models**:
  * Hollmann et al. (2023), [*TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*](https://arxiv.org/abs/2207.01848).
  * Hollmann et al. (2025), [*Accurate predictions on small data with a tabular foundation model*](https://www.nature.com/articles/s41586-024-08328-6).
  * Qu et al. (2025), [*TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*](https://proceedings.mlr.press/v267/qu25d.html).
  * Ma et al. (2025), [*TabDPT: Scaling Tabular Foundation Models on Real Data*](https://proceedings.neurips.cc/paper_files/paper/2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf).
  * NVIDIA (2026), [*NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction*](https://huggingface.co/blog/nvidia/kumo-tabular) — technical overview, accompanied by [model documentation](https://docs.nvidia.com/sdgm/structured-data-models/models/kumo-tabular).

* **Interpretable Additive Models**:
  * Agarwal et al. (2021), [*Neural Additive Models: Interpretable Machine Learning with Neural Nets*](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html).
  * Lengerich et al. (2020), [*Purifying Interaction Effects with the Functional ANOVA: An Efficient Algorithm for Recovering Identifiable Additive Models*](https://proceedings.mlr.press/v108/lengerich20a.html).
