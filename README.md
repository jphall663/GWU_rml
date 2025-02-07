## GWU_DNSC 6290: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, robustness, and security in machine learning.

* Lecture 1: Explainable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Bias Testing and Remediation
* Lecture 4: Machine Learning Security
* Lecture 5: Machine Learning Model Debugging
* Lecture 6: Responsible Machine Learning Best Practices
* Lecture 7: Risk Mitigation Proposals for Language Models

Corrections or suggestions? Please file a [GitHub issue](https://github.com/jphall663/GWU_rml/issues/new).

***

## Preliminary Materials
 * [Syllabus](https://docs.google.com/document/d/1msxf4_n9G3g5ejwvtE112FuW2a4UMo6b0ukgS8V0rA8/edit?usp=drive_link)
 * [Basic Data Manipulation](https://github.com/jphall663/GWU_data_mining/blob/master/01_basic_data_prep/01_basic_data_prep.md)
 * [Primer on Technical Malpractice](https://docs.google.com/presentation/d/1cZeaoIp4cQsVY_gj2a5Pg7ygexepQZRS-ZEn6n2QqEU/edit?usp=sharing)
 * [Whiteboard Notation](https://docs.google.com/presentation/d/1Axf9dizaE3XvGRelBHfsnAlMUPFuMExQ2WNVwQBKMrw/edit?usp=sharing)

## Lecture 1: Explainable Machine Learning Models

![Histogram, partial dependence, and ICE for a monotonic GBM and a credit card customer's most recent repayment status](/img/ebm.png)
<sub><sup>**Source:** [Simple Explainable Boosting Machine Example](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1_ebm_example.ipynb?flush_cache=true)</sup></sub>

### Lecture 1 Class Materials

* [Lecture Notes](tex/lecture_1.pdf)
* [Software Example](https://drive.google.com/file/d/1PnDSsNYRh1JNqZ3wyPgCA-KQxBop7y4L/view?usp=sharing)
* [Assignment 1](assignments/tex/assignment_1.pdf):
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb?flush_cache=true)
  * [Full evaluations results]()
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 2 (pp. 33 - 50) and Chapter 6 (pp. 189 - 217)
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 1 Additional Materials](additional_materials/am1.md)

***

## Lecture 2: Post-hoc Explanation

![A decision tree surrogate model forms a flow chart of a more complex monotonic GBM](/img/lecture_2.png)
<sub><sup>**Source:** [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)</sup></sub>

### Lecture 2 Class Materials

* [Lecture Notes](tex/lecture_2.pdf)
* [Software Example](https://colab.research.google.com/drive/1X7hagDcdMEU_YrGxsAXrUZo1hWCVN--H?usp=sharing)
* [Assignment 2](assignments/tex/assignment_2.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 2 (pp. 50 - 80) and Chapter 6 (pp. 208 - 230)
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 2 Additional Materials](additional_materials/am2.md)
  
***

## Lecture 3: Bias Testing and Remediation

![Two hundred neural networks from a random grid search trained on the UCI Credit Card Default dataset](/img/lecture_3.png)
<sub><sup>**Source:** [Lecture 3 Notes](tex/lecture_3.pdf)</sup></sub>

### Lecture 3 Class Materials

* [Lecture Notes](tex/lecture_3.pdf)
* [Assignment 3](assignments/tex/assignment_3.pdf)
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb?flush_cache=true)
  * [Full evaluations results](assignments/model_eval_2023_06_28_21_00_17.csv)
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
* Software Examples:
  * Sensitivity Analysis:
    * [Adversarial Example Search](https://colab.research.google.com/drive/1GBRrcZCoNJRYj5MI0iKKkXKUhg0vYUtB?usp=drive_link)
    * [Stress Testing](https://colab.research.google.com/drive/1S9pABlR7xs_VZAKraKT7pBSoXsuzZbaC?usp=sharing)
  * [Residual Analysis](https://colab.research.google.com/drive/1e8CXl23qpYsUL4nbEjX0MCsjhbHEPBVR?usp=sharing)  
* [Assignment 5](assignments/tex/assignment_5.pdf) 
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 3 and Chapter 8
* [Lecture 5 Additional Materials](additional_materials/am5.md)

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

## Lecture 7: Risk Mitigation Proposals for Language Models

![Past language model incidents.](/img/rag.png)

<sub><sup>A diagram for retrieval augmented generation. **Source:** [Lecture 7 notes](tex/lecture_7.pdf).</sup></sub>

### Lecture 7 Class Materials

* [Lecture Notes](tex/lecture_7.pdf)

### Lecture 7 Additional Tools

* [NIST AI RMF Retrieval Augmented Generation (RAG) Example](https://drive.google.com/drive/folders/1eR4iNqP2bbQHtnQx7Sj8_VhS_AeWPzBo?usp=sharing)
* [DAIR.AI Prompt Engineering Guide](https://www.promptingguide.ai)
* [PAI’s Responsible Practices for Synthetic Media](https://syntheticmedia.partnershiponai.org/)

### Lecture 7 Additional Reading

* **Introduction and Background**:
  * [_Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile_](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) 
  * [_Taxonomy of Risks posed by Language Models_](https://dl.acm.org/doi/pdf/10.1145/3531146.3533088)
  * [_On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?_](https://dl.acm.org/doi/10.1145/3442188.3445922)
