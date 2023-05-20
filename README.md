## GWU_DNSC 6290: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, security and privacy in machine learning.

* Lecture 1: Explainable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Discrimination Testing and Remediation
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
  * [Syllabus](https://github.com/jphall663/GWU_rml/blob/master/tentative_syllabus_ph_responsible_machine_learning_msba_v5.pdf)
  * [Basic Data Manipulation](https://github.com/jphall663/GWU_data_mining/blob/master/01_basic_data_prep/01_basic_data_prep.md)
  * [Primer on Technical Malpractice](https://docs.google.com/presentation/d/1cZeaoIp4cQsVY_gj2a5Pg7ygexepQZRS-ZEn6n2QqEU/edit?usp=sharing)
  * [Whiteboard Notation](https://docs.google.com/presentation/d/1Axf9dizaE3XvGRelBHfsnAlMUPFuMExQ2WNVwQBKMrw/edit?usp=sharing)
* [Lecture Notes](tex/lecture_1.pdf)
* [Assignment 1](assignments/tex/assignment_1.pdf): 
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb)
  * [Full evaluations results]()
* Reading: _Machine Learning for High-Risk Applications_, Chapter 2 (pp. 33 - 50) and Chapter 6 (pp. 189 - 217)

### Lecture 1 Additional Software Tools

* Python [PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox)
* Python [explainable boosting machine (EBM)/GA2M](https://github.com/interpretml/interpret)
* R [`gam`](https://cran.r-project.org/web/packages/gam/index.html)
* `h2o` [penalized GLM](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html) (R and Python)
* Monotonic gradient boosting machine (GBM): [`h2o`](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/monotone_constraints.html) and [`xgboost`](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html) (R and Python)
* R [`rpart`](https://cran.r-project.org/web/packages/rpart/index.html)
* Python [`skope-rules`](https://github.com/scikit-learn-contrib/skope-rules)

### Lecture 1 Additional Software Examples

* [Building from Penalized GLM to Monotonic GBM (simple)](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1.ipynb?flush_cache=true)
* [Building from Penalized GLM to Monotonic GBM](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/glm_mgbm_gbm.ipynb?flush_cache=true)
* [Simple Explainable Boosting Machine Example](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1_ebm_example.ipynb?flush_cache=true)
* [PiML Assignment 1 Example](https://github.com/jphall663/GWU_rml/blob/master/assignments/assignment_1/group6_PiML_example.ipynb) and simple [requirements.txt](https://github.com/jphall663/GWU_rml/blob/master/assignments/assignment_1/piml_requirements.txt)

### Lecture 1 Additional Reading

* **Introduction and Background**: 
  * [*An Introduction to Machine Learning Interpretability*](https://h2o.ai/content/dam/h2o/en/marketing/documents/2019/08/An-Introduction-to-Machine-Learning-Interpretability-Second-Edition.pdf) 
  * [*Designing Inherently Interpretable Machine Learning Models*](https://arxiv.org/pdf/2111.01743.pdf)
  * [*Psychological Foundations of Explainability and Interpretability in Artificial Intelligence*](https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8367.pdf)
  * [*Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*](https://arxiv.org/pdf/1811.10154.pdf) 

* **Self-explainaing Machine Learning Techniques**:
  * [*Accurate Intelligible Models with Pairwise Interactions*](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
  * **Elements of Statistical Learning** - Chapters 
  * **Interpretable Machine Learning** - [Chapter 5](https://christophm.github.io/interpretable-ml-book/simple.html)
  * [*This Looks Like That: Deep Learning for Interpretable Image Recognition*](https://arxiv.org/pdf/1806.10574.pdf)

FIGS: Fast Interpretable Greedy-Tree Sums (Tan, et al. 2022)
XGB1: Extreme Gradient Boosted Trees of Depth 1, with optimal binning (Chen and Guestrin, 2016; Navas-Palencia, 2020)
XGB2: Extreme Gradient Boosted Trees of Depth 2, with effect purification (Chen and Guestrin, 2016; Lengerich, et al. 2020)
EBM: Explainable Boosting Machine (Nori, et al. 2019; Lou, et al. 2013)
GAMI-Net: Generalized Additive Model with Structured Interactions (Yang, Zhang and Sudjianto, 2021)
ReLU-DNN: Deep ReLU Networks using Aletheia Unwrapper and Sparsification (Sudjianto, et al. 2020)

***

## Lecture 2: Post-hoc Explanation

![A decision tree surrogate model forms a flow chart of a more complex monotonic GBM](/img/lecture_2.png)
<sub><sup>**Source:** [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)</sup></sub>

### Lecture 2 Class Materials

* [Lecture Notes](tex/lecture_2.pdf) 
* Software Examples: 
  * [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb) 
  * [Building from Penalized GLM to Monotonic GBM](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/glm_mgbm_gbm.ipynb?flush_cache=true)
  * [Monotonic XGBoost models, partial dependence, individual conditional expectation plots, and Shapley explanations](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/xgboost_pdp_ice.ipynb)
  * [Decision tree surrogates, LOCO, and ensembles of explanations](https://nbviewer.org/github/jphall663/interpretable_machine_learning_with_python/blob/master/dt_surrogate_loco.ipynb)
* [Assignment 2](https://raw.githubusercontent.com/jphall663/GWU_rml/master/assignments/tex/assignment_2.pdf) 

### Lecture 2 Suggested Software

* Python:
  * [`allennlp`](https://github.com/allenai/allennlp)
  * [`alibi`](https://github.com/SeldonIO/alibi)
  * [`interpret`](https://github.com/interpretml/interpret)
  * [`shap`](https://github.com/slundberg/shap)
  * [`PiML-Toolbox`](https://github.com/SelfExplainML/PiML-Toolbox)
  * [`tf-explain`](https://github.com/sicara/tf-explain)
* R:
  * [`DALEX`](https://cran.r-project.org/web/packages/DALEX/index.html)
  * [`iml`](https://cran.r-project.org/web/packages/iml/index.html)
  
### Lecture 2 Suggested Reading

* **Introduction and Background**: 
  * [*Proposed Guidelines for the Responsible Use of Explainable Machine Learning*](https://arxiv.org/pdf/1906.03533.pdf)
  * [**Responsible Artificial Intelligence**](https://www.springer.com/gp/book/9783030303709) - Chapter 4
  
* **Post-hoc Explanation Techniques**:
  * **Interpretable Machine Learning** - [Chapter 6](https://christophm.github.io/interpretable-ml-book/agnostic.html) and [Chapter 7](https://christophm.github.io/interpretable-ml-book/example-based.html)
  * [*Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks*](https://arxiv.org/pdf/1711.06104.pdf)

***

## Lecture 3: Discrimination Testing and Remediation

![Two hundred neural networks from a random grid search trained on the UCI Credit Card Default dataset](/img/lecture_3.png)
<sub><sup>**Source:** [Lecture 3 Notes](tex/lecture_3.pdf)</sup></sub>

### Lecture 3 Class Materials

* [Lecture Notes](tex/lecture_3.pdf) 
* Software Example: [Testing a Constrained Model for Discrimination and Remediating Discovered Discrimination](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_3.ipynb) 
* [Assignment 3](https://raw.githubusercontent.com/jphall663/GWU_rml/master/assignments/tex/assignment_3.pdf) 

### Lecture 3 Suggested Software

Python:

  * [`aequitas`](https://github.com/dssg/aequitas)
  * [`AIF360`](https://github.com/IBM/AIF360)
  * [`Themis`](https://github.com/LASER-UMASS/Themis)
  
### Lecture 3 Suggested Reading

* **Introduction and Background**:

  * **Fairness and Machine Learning** - [Introduction](https://fairmlbook.org/introduction.html)
  * [NIST SP1270: _Towards a Standard for Identifying and Managing Bias in Artificial Intelligence_](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.1270.pdf)
  * [*Fairness Through Awareness*](https://arxiv.org/pdf/1104.3913.pdf)
  
* **Discrimination Testing and Remediation Techniques**:

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
* Software Example: [Attacking a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_4.ipynb) 
* [Assignment 4](https://raw.githubusercontent.com/jphall663/GWU_rml/master/assignments/tex/assignment_4.pdf) 

### Lecture 4 Suggested Software

Python:

  * [`cleverhans`](https://github.com/tensorflow/cleverhans)
  * [`foolbox`](https://github.com/bethgelab/foolbox)
  * [`robustness`](https://github.com/MadryLab/robustness)
  
### Lecture 4 Suggested Reading

* **Introduction and Background**:

  * [*A Marauder’s Map of Security and Privacy in Machine Learning*](https://arxiv.org/pdf/1811.01134.pdf)
  * [BIML Interactive Machine Learning Risk Framework](https://berryvilleiml.com/interactive/)
  * [*The Security of Machine Learning*](https://people.eecs.berkeley.edu/~adj/publications/paper-files/SecML-MLJ2010.pdf)
  * [*Proposals for model vulnerability and security*](https://www.oreilly.com/content/proposals-for-model-vulnerability-and-security/)

  
* **Machine Learning Attacks**:

  * [*Membership Inference Attacks Against Machine Learning Models*](https://arxiv.org/pdf/1610.05820.pdf)
  * [*Stealing Machine Learning Models via Prediction APIs*](https://arxiv.org/pdf/1609.02943.pdf)
  * [*Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures*](https://dl.acm.org/doi/pdf/10.1145/2810103.2813677)
  * [*Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers*](https://arxiv.org/pdf/1306.4447.pdf)

* **Links from Lecture 4**:
  * [Mitre Adversarial Threat Matrix](https://github.com/mitre/advmlthreatmatrix)
  * [Robust ML](https://www.robust-ml.org/)   
  * [ISIS 'still evading detection on Facebook', report says](https://www.bbc.com/news/technology-53389657) 
  * [Researchers bypass airport and payment facial recognition systems using masks](https://www.engadget.com/2019-12-16-facial-recognition-fooled-masks.html)
  * [Slight Street Sign Modifications Can Completely Fool Machine Learning Algorithms](https://spectrum.ieee.org/cars-that-think/transportation/sensors/slight-street-sign-modifications-can-fool-machine-learning-algorithms)

***   

## Lecture 5: Machine Learning Model Debugging

![Residuals for an important feature betray a serious problem in a machine learning model.](img/lecture_5.png)
<sub><sup>**Source:** [Real-World Strategies for Model Debugging](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)</sup></sub>

### Lecture 5 Class Materials

* [Lecture Notes](tex/lecture_5.pdf)
* Software Example: [Debugging a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_5.ipynb) 
* [Assignment 5](https://raw.githubusercontent.com/jphall663/GWU_rml/master/assignments/tex/assignment_5.pdf)

### Lecture 5 Suggested Software

* [`DALEX`](https://cran.r-project.org/web/packages/DALEX/index.html) (R)
* [`SALib`](https://github.com/SALib/SALib) (Python)
* [`themis-ml`](https://github.com/cosmicBboy/themis-ml) (Python)
* [`What-if Tool`](https://pair-code.github.io/what-if-tool/index.html) (Browser)
  
  
### Lecture 5 Suggested Reading

* **Introduction and Background**:

  * [Debugging Machine Learning Models](https://debug-ml-iclr2019.github.io/)
  * [*Why you should care about debugging machine learning models*](https://www.oreilly.com/radar/why-you-should-care-about-debugging-machine-learning-models/)
  * [*Real-World Strategies for Model Debugging*](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)

* **Links from Lecture 5**:
  
  * [AI Incident Tracker](https://github.com/jphall663/awesome-machine-learning-interpretability/blob/master/README.md#ai-incident-tracker)
  * [AI Incident Database](https://incidentdatabase.ai/)
  * [Debugging Machine Learning Models](https://debug-ml-iclr2019.github.io/)
  * [_Underspecification Presents Challenges for Credibility in Modern Machine Learning_](https://arxiv.org/pdf/2011.03395.pdf)
  
***   

## Lecture 6: Responsible Machine Learning Best Practices 
 
![A responsible machine learning workingflow](/img/rml_diagram_no_hilite.png)

<sub><sup>A Responsible Machine Learning Workflow Diagram. **Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3).</sup></sub>

### Lecture 6 Class Materials

* [Lecture Notes](tex/lecture_6.pdf)
* [Assignment 6 (Final Assessment)](assignments/tex/assignment_6.pdf)

### Lecture 6 Suggested Software

* [Awesome Machine Learning Interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)

### Lecture 6 Suggested Reading

* **Introduction and Background**:

  * [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
  * [Interagency Guidance on Model Risk Management (SR 11-7)](https://www.federalreserve.gov/supervisionreg/srletters/sr1107a1.pdf)
  * [Eight Principles of Responsible Machine Learning](https://ethical.institute/principles.html)
  * [Principles for Accountable Algorithms and a Social Impact Statement for Algorithms](https://www.fatml.org/resources/principles-for-accountable-algorithms)
  * [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
    
* **Links from Lecture 6**:

  * [_Predicting Good Probabilities With Supervised Learning_](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)


***   

## Using Class Software Resources

### Mac

**Preliminaries:**

* Install [`homebrew`](https://brew.sh/): `$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"`

* Install Java:

```
brew tap adoptopenjdk/openjdk
brew cask install adoptopenjdk8
```

* [Install Mac command line developer tools](https://www.cnet.com/how-to/install-command-line-developer-tools-in-os-x/)

* Install Graphviz: `$ brew install graphviz`

* Install Git: `$ brew install git`

1. Clone this repository with the examples: `$ git clone https://github.com/jphall663/GWU_rml.git`

2. Install Anaconda Python 5.1.0 from the [Anaconda archives](https://repo.continuum.io/archive/) (Anaconda3-5.1.0-MacOSX-x86_64.pkg) and *add it to your system path.* (This means when you type `$ python` at the command prompt, you will start a Python 3.6 session. You should probably check for a line like this `export PATH="/Users/phall/anaconda/bin:$PATH"` in your `.bash_profile` file. You may also need to close and restart your terminal session.)

3. Install `virtualenv`: `$ pip install virtualenv` 
 
4. Change directories into the cloned repository: `$ cd GWU_rml`

5. Create a Python 3.6 virtual environment: `$ virtualenv -p /path/to/anaconda3/bin/python3.6 env_rml` (`/path/to/anaconda3/bin/python3.6` is not a real path. You must replace it with the path to `python3.6` on your system. This can be as simple as `$ virtualenv -p python3.6 env_rml`.) 

6. Activate the virtual environment: `$ source env_rml/bin/activate`

7. Install the correct packages for the example notebooks: `$ pip install -r requirements.txt`

8. Start Jupyter: `$ jupyter notebook`

9. When you are finished, deactivate the environment: `$ deactivate`

If you get stuck, try this link: [Mac Setup: Virtualenv](https://sourabhbajaj.com/mac-setup/Python/virtualenv.html).

### Ubuntu

1. Install [Git](https://git-scm.com/downloads) (`$ sudo apt-get install git`), [Graphviz](https://www.graphviz.org/) (`$ sudo apt-get install graphviz`) and Java (`$ sudo apt-get install openjdk-8-jdk`). They must all be *added to your system path.*

2. Clone this repository with the examples: `$ git clone https://github.com/jphall663/GWU_rml.git`

3. Install Anaconda Python 5.1.0 from the [Anaconda archives](https://repo.continuum.io/archive/) (Anaconda3-5.1.0-Linux-x86_64.sh) and *add it to your system path.* (This means when you type `$ python` at the command prompt, you will start a Python 3.6 session.)

4. Install `virtualenv`: `$ pip install virtualenv` 

5. Change directories into the cloned repository: `$ cd GWU_rml`

6. Create a Python 3.6 virtual environment: `$ virtualenv -p /path/to/anaconda3/bin/python3.6 env_rml` 

7. Activate the virtual environment: `$ source env_rml/bin/activate`

8. Install the correct packages for the example notebooks: `$ pip install -r requirements.txt`

9. Start Jupyter: `$ jupyter notebook`

10. When you are finished, deactivate the environment: `$ deactivate`.

***

Some materials Copyright Patrick Hall and the H2O.ai team 2017-2020.  
