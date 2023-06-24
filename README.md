## GWU_DNSC 6290: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, robustness, and security in machine learning.

* Lecture 1: Explainable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Bias Testing and Remediation
* Lecture 4: Machine Learning Security
* Lecture 5: Machine Learning Model Debugging
* Lecture 6: Responsible Machine Learning Best Practices

Corrections or suggestions? Please file a [GitHub issue](https://github.com/jphall663/GWU_rml/issues/new).

***

## Lecture 1: Explainable Machine Learning Models

![Histogram, partial dependence, and ICE for a monotonic GBM and a credit card customer's most recent repayment status](/img/ebm.png)
<sub><sup>**Source:** [Simple Explainable Boosting Machine Example](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1_ebm_example.ipynb?flush_cache=true)</sup></sub>

### Lecture 1 Class Materials

* Introduction:
  * [Syllabus](https://github.com/jphall663/GWU_rml/blob/master/syllabus_ph_responsible_machine_learning_msba_v5.1.pdf)
  * [Basic Data Manipulation](https://github.com/jphall663/GWU_data_mining/blob/master/01_basic_data_prep/01_basic_data_prep.md)
  * [Primer on Technical Malpractice](https://docs.google.com/presentation/d/1cZeaoIp4cQsVY_gj2a5Pg7ygexepQZRS-ZEn6n2QqEU/edit?usp=sharing)
  * [Whiteboard Notation](https://docs.google.com/presentation/d/1Axf9dizaE3XvGRelBHfsnAlMUPFuMExQ2WNVwQBKMrw/edit?usp=sharing)
* [Lecture Notes](tex/lecture_1.pdf)
* [Assignment 1](assignments/tex/assignment_1.pdf):
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb?flush_cache=true)
  * [Full evaluations results](assignments/model_eval_2023_06_21_12_52_47.csv)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 2 (pp. 33 - 50) and Chapter 6 (pp. 189 - 217)

### Lecture 1 Additional Software Tools

* **Python**:
  * [causalml](https://oreil.ly/XsiMk)
  * [interpret](https://github.com/interpretml/interpret)
  * [imodels](https://oreil.ly/coPjR)
  * [PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox)
  * [sklearn-expertsys](https://oreil.ly/igFz6)
  * [skope-rules](https://github.com/scikit-learn-contrib/skope-rules)
  * [tensorflow/lattice](https://oreil.ly/Z9iCS)

* **R**:
  * [arules](https://oreil.ly/bBv9s)
  * [elasticnet](https://oreil.ly/pBOBN)
  * [gam](https://cran.r-project.org/web/packages/gam/index.html)
  * [glmnet](https://oreil.ly/rMzEl)
  * [quantreg](https://oreil.ly/qBWk9)
  * [rpart](https://cran.r-project.org/web/packages/rpart/index.html)
  * [RuleFit](https://oreil.ly/K-qc4)

* **Python, R or other**:
  * [h2o-3](https://oreil.ly/PPUk5)
  * [Rudin Group code](https://oreil.ly/QmRFF)
  * [xgboost](https://github.com/dmlc/xgboost)

### Lecture 1 Additional Software Examples

* [Building from Penalized GLM to Monotonic GBM (simple)](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1.ipynb?flush_cache=true)
* [Building from Penalized GLM to Monotonic GBM](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/glm_mgbm_gbm.ipynb?flush_cache=true)
* [Simple Explainable Boosting Machine Example](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1_ebm_example.ipynb?flush_cache=true)
* [PiML Assignment 1 Example](https://github.com/jphall663/GWU_rml/blob/master/assignments/assignment_1/group6_PiML_example.ipynb) and simple [requirements.txt](https://github.com/jphall663/GWU_rml/blob/master/assignments/assignment_1/piml_requirements.txt)
* _Machine Learning for High-risk Applications_: [Use Cases](https://oreil.ly/machine-learning-high-risk-apps-code) (Chapter 6)

### Lecture 1 Additional Reading

* **Introduction and Background**:
  * [*An Introduction to Machine Learning Interpretability*](https://h2o.ai/content/dam/h2o/en/marketing/documents/2019/08/An-Introduction-to-Machine-Learning-Interpretability-Second-Edition.pdf)
  * [*Designing Inherently Interpretable Machine Learning Models*](https://arxiv.org/pdf/2111.01743.pdf)
  * [*Psychological Foundations of Explainability and Interpretability in Artificial Intelligence*](https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8367.pdf)
  * [*Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*](https://arxiv.org/pdf/1811.10154.pdf)

* **Explainable Machine Learning Techniques**:
  * [*Accurate Intelligible Models with Pairwise Interactions*](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
  * **Elements of Statistical Learning** - Chapters 3,4, and 9
  * [*Fast Interpretable Greedy-Tree Sums (FIGS)*](https://arxiv.org/pdf/2201.11931.pdf)
  * **Interpretable Machine Learning** - [Chapter 5](https://christophm.github.io/interpretable-ml-book/simple.html)
  * [*GAMI-Net: An Explainable Neural Network Based on Generalized Additive Models with Structured Interactions*](https://www.sciencedirect.com/science/article/abs/pii/S0031320321003484)
  * [*Neural Additive Models: Interpretable Machine Learning with Neural Nets*](https://proceedings.neurips.cc/paper_files/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf)
  * [*A Responsible Machine Learning Workflow with Focus on Interpretable Models, Post-hoc Explanation, and Discrimination Testing*](https://www.mdpi.com/2078-2489/11/3/137)
  * [*This Looks Like That: Deep Learning for Interpretable Image Recognition*](https://arxiv.org/pdf/1806.10574.pdf)
  * [*Unwrapping The Black Box of Deep ReLU Networks: Interpretability, Diagnostics, and Simplification*](https://arxiv.org/pdf/2011.04041.pdf)

***

## Lecture 2: Post-hoc Explanation

![A decision tree surrogate model forms a flow chart of a more complex monotonic GBM](/img/lecture_2.png)
<sub><sup>**Source:** [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)</sup></sub>

### Lecture 2 Class Materials

* [Lecture Notes](tex/lecture_2.pdf)
* [Assignment 2](assignments/tex/assignment_2.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 2 (pp. 50 - 80) and Chapter 6 (pp. 208 - 230)

### Lecture 2 Additional Software Tools

* **Python**:
  * [allennlp](https://github.com/allenai/allennlp)
  * [alibi](https://github.com/SeldonIO/alibi)
  * [anchor](https://oreil.ly/K3UuW)
  * [DiCE](https://oreil.ly/-lwV4)
  * [interpret](https://github.com/interpretml/interpret)
  * [lime](https://oreil.ly/j5Cqj)
  * [shap](https://github.com/slundberg/shap)
  * [PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox)
  * [tf-explain](https://github.com/sicara/tf-explain)

* **R**:
  * [ALEPlot](https://oreil.ly/OSfUT)
  * [DALEX](https://cran.r-project.org/web/packages/DALEX/index.html)
  * [ICEbox](https://oreil.ly/6nl1W)
  * [iml](https://cran.r-project.org/web/packages/iml/index.html)
  * [Model Oriented](https://oreil.ly/7wUMp)
  * [pdp](https://oreil.ly/PasMQ)
  * [shapFlex](https://oreil.ly/RADtC)
  * [vip](https://oreil.ly/YcD2_)

* **Python, R or other**:
  * [h2o-3](https://oreil.ly/GtGvK)

### Lecture 2 Additional Software Examples
  * [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)
  * [Building from Penalized GLM to Monotonic GBM](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/glm_mgbm_gbm.ipynb?flush_cache=true)
  * [Monotonic XGBoost models, partial dependence, individual conditional expectation plots, and Shapley explanations](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/xgboost_pdp_ice.ipynb)
  * [Decision tree surrogates, LOCO, and ensembles of explanations](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/dt_surrogate_loco.ipynb)
  * _Machine Learning for High-risk Applications_: [Use Cases](https://oreil.ly/machine-learning-high-risk-apps-code) (Chapter 6)

### Lecture 2 Additional Reading

* **Introduction and Background**:
  * [*On the Art and Science of Explainable Machine Learning*](https://oreil.ly/myVr8)
  * [*Proposed Guidelines for the Responsible Use of Explainable Machine Learning*](https://arxiv.org/pdf/1906.03533.pdf)

* **Post-hoc Explanation Techniques**:
  * [_A Unified Approach to Interpreting Model Predictions_](https://papers.nips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
  * [_Anchors: High-Precision Model-Agnostic Explanations_](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)
  * **Elements of Statistical Learning** - [Section 10.13](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf)
  * [_Extracting Tree-Structured Representations of Trained Networks_](https://proceedings.neurips.cc/paper/1995/file/45f31d16b1058d586fc3be7207b58053-Paper.pdf)
  * [_Interpretability via Model Extraction_](https://arxiv.org/pdf/1706.09773.pdf)
  * **Interpretable Machine Learning** - [Chapter 6](https://christophm.github.io/interpretable-ml-book/agnostic.html) and [Chapter 7](https://christophm.github.io/interpretable-ml-book/example-based.html)
  * [_Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation_](https://arxiv.org/pdf/1309.6392.pdf)
  * [*Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks*](https://arxiv.org/pdf/1711.06104.pdf)
  * [_Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models_](https://arxiv.org/pdf/1612.08468.pdf)
  * [_“Why Should I Trust You?” Explaining the Predictions of Any Classifier_](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)

* **Problems with Post-hoc Explanation**:
  * [*General Pitfalls of Model-Agnostic Interpretation Methods*](https://oreil.ly/On9uS)
  * [_Limitations of Interpretable Machine Learning Methods_](https://oreil.ly/VHMWh)
  * [*When Not to Trust Your Explanations*](https://oreil.ly/9Oxa6)

***

## Lecture 3: Bias Testing and Remediation

![Two hundred neural networks from a random grid search trained on the UCI Credit Card Default dataset](/img/lecture_3.png)
<sub><sup>**Source:** [Lecture 3 Notes](tex/lecture_3.pdf)</sup></sub>

### Lecture 3 Class Materials

* [Lecture Notes](tex/lecture_3.pdf)
* [Assignment 3](assignments/tex/assignment_3.pdf)
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb?flush_cache=true)
  * [Full evaluations results](assignments/model_eval_2023_06_21_12_52_47.csv)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 4 and Chapter 10

### Lecture 3 Additional Software Tools

* **Python**:
  * [aequitas](https://github.com/dssg/aequitas)
  * [AIF360](https://github.com/IBM/AIF360)
  * [Algorithmic Fairness](https://oreil.ly/JNzqk)
  * [fairlearn](https://oreil.ly/jYjCi)
  * [fairml](https://oreil.ly/DCkZ5)
  * [solas-ai-disparity](https://oreil.ly/X9fd6)
  * [tensorflow/fairness-indicators](https://oreil.ly/dHBSL)
  * [Themis](https://github.com/LASER-UMASS/Themis)

* **R**:
  * [AIF360](https://oreil.ly/J53bZ)
  * [fairmodels](https://oreil.ly/nSv8B)
  * [fairness](https://oreil.ly/Dequ9)

### Lecture 3 Additional Software Examples
* [Increase Fairness in Your Machine Learning Project with Disparate Impact Analysis using Python and H2O](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/dia.ipynb)
* [Testing a Constrained Model for Discrimination and Remediating Discovered Discrimination](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_3.ipynb)
* _Machine Learning for High-risk Applications_: [Use Cases](https://oreil.ly/machine-learning-high-risk-apps-code) (Chapter 10)

### Lecture 3 Additional Reading

* **Introduction and Background**:
  * [*50 Years of Test (Un)fairness: Lessons for Machine Learning*](https://oreil.ly/fTlda)
  * **Fairness and Machine Learning** - [Introduction](https://fairmlbook.org/introduction.html)
  * [NIST SP1270: _Towards a Standard for Identifying and Managing Bias in Artificial Intelligence_](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf)
  * [*Fairness Through Awareness*](https://arxiv.org/pdf/1104.3913.pdf)

* **Discrimination Testing and Remediation Techniques**:
  * [*An Empirical Comparison of Bias Reduction Methods on Real-World Problems in High-Stakes Policy Settings*](https://oreil.ly/vmxPz)
  * [*Certifying and Removing Disparate Impact*](https://arxiv.org/pdf/1412.3756.pdf)
  * [*Data Preprocessing Techniques for Classification Without
Discrimination*](https://link.springer.com/content/pdf/10.1007/s10115-011-0463-8.pdf)  
  * [*Decision Theory for Discrimination-aware Classification*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.722.3030&rep=rep1&type=pdf)
  * [*Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification Without Disparate Mistreatment*](https://arxiv.org/pdf/1610.08452.pdf)
  * [*Learning Fair Representations*](http://proceedings.mlr.press/v28/zemel13.pdf)
  * [*Mitigating Unwanted Biases with Adversarial Learning*](https://dl.acm.org/doi/pdf/10.1145/3278721.3278779)

***   

## Lecture 4: Machine Learning Security

![A cheatsheet for ML attacks](img/Attack_Cheat_Sheet.png)
<sub><sup>**Source:** [Responsible Machine Learning](https://resources.oreilly.com/examples/0636920415947/blob/master/Attack_Cheat_Sheet.png)</sup></sub>

### Lecture 4 Class Materials

* [Lecture Notes](tex/lecture_4.pdf)
* [Assignment 4](assignments/tex/assignment_4.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 5 and Chapter 11

### Lecture 4 Additional Software Tools

* [adversarial-robustness-toolbox](https://oreil.ly/5eXYi)
* [counterfit](https://oreil.ly/4WM4P)
* [cleverhans](https://github.com/tensorflow/cleverhans)
* [foolbox](https://github.com/bethgelab/foolbox)
* [ml_privacy_meter](https://oreil.ly/HuHxf)
* [NIST de-identification tools](https://oreil.ly/M8xhr)
* [robustness](https://github.com/MadryLab/robustness)
* [tensorflow/privacy](https://oreil.ly/hkurv)

#### Lecture 4 Additional Software Examples

* [Attacking a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_4.ipynb)
* _Machine Learning for High-risk Applications_: [Use Cases](https://oreil.ly/machine-learning-high-risk-apps-code) (Chapter 11)

### Lecture 4 Additional Reading

* **Introduction and Background**:

  * [*A Marauder’s Map of Security and Privacy in Machine Learning*](https://arxiv.org/pdf/1811.01134.pdf)
  * [BIML Interactive Machine Learning Risk Framework](https://berryvilleiml.com/interactive/)
  * [FTC's "Start with Security" guidelines](https://oreil.ly/jmeja)
  * [Mitre Adversarial Threat Matrix](https://github.com/mitre/advmlthreatmatrix)
  * [NIST Computer Security Resource Center](https://oreil.ly/pncXb)
  * [*The Security of Machine Learning*](https://people.eecs.berkeley.edu/~adj/publications/paper-files/SecML-MLJ2010.pdf)
  * [*Proposals for model vulnerability and security*](https://www.oreilly.com/content/proposals-for-model-vulnerability-and-security/)

* **Machine Learning Attacks and Countermeasures**:

  * [*Membership Inference Attacks Against Machine Learning Models*](https://arxiv.org/pdf/1610.05820.pdf)
  * [*Stealing Machine Learning Models via Prediction APIs*](https://arxiv.org/pdf/1609.02943.pdf)
  * [*Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677)
  * [*Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers*](https://arxiv.org/pdf/1306.4447.pdf)
  * [Robust ML](https://www.robust-ml.org/)  
  * [*Sponge Examples: Energy-latency Attacks on Neural Networks*](https://arxiv.org/pdf/2006.03463.pdf)

* **Examples of Real-world Attacks**:

  * [Fraudsters Cloned Company Director’s Voice In $35 Million Heist, Police Find](https://www.forbes.com/sites/thomasbrewster/2021/10/14/huge-bank-fraud-uses-deep-fake-voice-tech-to-steal-millions/?sh=7f3ba4bd7559)
  * [ISIS 'still evading detection on Facebook', report says](https://www.bbc.com/news/technology-53389657)
  * [Researchers bypass airport and payment facial recognition systems using masks](https://www.engadget.com/2019-12-16-facial-recognition-fooled-masks.html)
  * [Slight Street Sign Modifications Can Completely Fool Machine Learning Algorithms](https://spectrum.ieee.org/cars-that-think/transportation/sensors/slight-street-sign-modifications-can-fool-machine-learning-algorithms)
  * [These students figured out their tests were graded by AI — and the easy way to cheat](https://www.theverge.com/2020/9/2/21419012/edgenuity-online-class-ai-grading-keyword-mashing-students-school-cheating-algorithm-glitch)

***   

## Lecture 5: Machine Learning Model Debugging

![Residuals for an important feature betray a serious problem in a machine learning model.](img/lecture_5.png)
<sub><sup>**Source:** [Real-World Strategies for Model Debugging](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)</sup></sub>

### Lecture 5 Class Materials

* [Lecture Notes](tex/lecture_5.pdf)
* [Assignment 5](assignments/tex/assignment_5.pdf) 
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 3 and Chapter 8

### Lecture 5 Additional Software Tools

* **Python**:
  * [mlextend](https://oreil.ly/j27C_)
  * [PiML](https://oreil.ly/7QLK1)
  * [SALib](https://oreil.ly/djeTQ)
  * [themis-ml](https://github.com/cosmicBboy/themis-ml)

* **R**:
  * [DALEX](https://cran.r-project.org/web/packages/DALEX/index.html) 
  * [drifter](https://oreil.ly/Pur4F)

* **Other**:
  * [manifold](https://oreil.ly/If0n5)
  * [What-If Tool](https://oreil.ly/1n-Fl)

### Lecture 5 Additional Software Examples

* [Advanced residual analysis example](https://oreil.ly/Poe20)
* [Advanced sensitivity analysis example](https://oreil.ly/QPFFx)
* [Basic sensitivity and residual analysis example](https://oreil.ly/Tcu65)
* [Debugging a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_5.ipynb)
* _Machine Learning for High-risk Applications_: [Use Cases](https://oreil.ly/machine-learning-high-risk-apps-code) (Chapter 8)

### Lecture 5 Additional Reading

* **Introduction and Background**:

  * [AI Incident Database](https://incidentdatabase.ai/)
  * [Debugging Machine Learning Models](https://debug-ml-iclr2019.github.io/)
  * [*Overview of Debugging ML Models*](https://oreil.ly/xZGoN)
  * [*Real-World Strategies for Model Debugging*](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)
  * [*Safe and Reliable Machine Learning*](https://oreil.ly/mLU8l)
  * [*Why you should care about debugging machine learning models*](https://www.oreilly.com/radar/why-you-should-care-about-debugging-machine-learning-models/)
   
* **Debugging Approaches and Information**:

  * [*A Comprehensive Study on Deep Learning Bug Characteristics*](https://oreil.ly/89R6O)
  * [*DQI: Measuring Data Quality in NLP*](https://oreil.ly/aa7rv)
  * [*Identifying and Overcoming Common Data Mining Mistakes*](https://oreil.ly/w19Qm)
  * [PiML User Guide: Diagnostic Suite](https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/testing.html)
  * [_Predicting Good Probabilities With Supervised Learning_](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
  * [_Underspecification Presents Challenges for Credibility in Modern Machine Learning_](https://arxiv.org/pdf/2011.03395.pdf)

***   

## Lecture 6: Responsible Machine Learning Best Practices

![A responsible machine learning workingflow](/img/rml_diagram_no_hilite.png)

<sub><sup>A Responsible Machine Learning Workflow Diagram. **Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3).</sup></sub>

### Lecture 6 Class Materials

* [Lecture Notes](tex/lecture_6.pdf)
* [Assignment 6 (Final Assessment)](assignments/tex/assignment_6.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 1 and Chapter 12

### Lecture 6 Additional Software Tools and Examples

* [Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)

### Lecture 6 Additional Reading

* **Introduction and Background**:
  * [*A Responsible Machine Learning Workflow with Focus on Interpretable Models, Post-hoc Explanation, and Discrimination Testing*](https://www.mdpi.com/2078-2489/11/3/137)
  * [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
  * [Interagency Guidance on Model Risk Management (SR 11-7)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf)
  * [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
