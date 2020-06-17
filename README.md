![A responsible machine learning workingflow](/img/rml_diagram_no_hilite.png)

<sub><sup>A Responsible Machine Learning Workflow Diagram. **Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3).</sup></sub>

## GWU_DNSC 6290: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, security and privacy in machine learning.

* Lecture 1: Interpretable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Discrimination Testing and Remediation
* Lecture 4: Machine Learning Security
* Lecture 5: Machine Learning Model Debugging
* Lecture 6: Responsible Machine Learning Best Practices

Corrections or suggestions? Please file a [GitHub issue](https://github.com/jphall663/GWU_rml/issues/new).

***

## Lecture 1: Interpretable Machine Learning Models

![Histogram, partial dependence, and ICE for a monotonic GBM and a credit card customer's most recent repayment status](/img/lecture_1.png)
<sub><sup>**Source:** [Building from Penalized GLM to Monotonic GBM](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1.ipynb)</sup></sub>

### Lecture 1 Class Materials

* [Syllabus](rml_syllabus_summer_2020.pdf)
* [Lecture Notes](tex/lecture_1.pdf)
* [Lecture Video](https://www.youtube.com/watch?v=AJS1_2DIbjo)
* Software Example: [Building from Penalized GLM to Monotonic GBM](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1.ipynb)

### Lecture 1 Suggested Software

* Python [explainable boosting machine (EBM)/GA2M](https://github.com/interpretml/interpret)
* R [`gam`](https://cran.r-project.org/web/packages/gam/index.html)
* `h2o` [penalized GLM](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html) (R and Python)
* Monotonic gradient boosting machine (GBM): [`h2o`](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/monotone_constraints.html) and [`xgboost`](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html) (R and Python)
* R [`rpart`](https://cran.r-project.org/web/packages/rpart/index.html)
* Python [`skope-rules`](https://github.com/scikit-learn-contrib/skope-rules)

### Lecture 1 Suggested Reading

* **Introduction and Background**: 
  * [*Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead*](https://www.nature.com/articles/s42256-019-0048-x) 
  * **[Responsible Artificial Intelligence](https://www.springer.com/gp/book/9783030303709)** - Sections 2.1-2.5, Chapter 7
  
* **Interpretable Machine Learning Techniques**:
  * **Interpretable Machine Learning** - [Chapter 4](https://christophm.github.io/interpretable-ml-book/simple.html)
  * [*Accurate Intelligible Models with Pairwise Interactions*](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf)
  * [*This Looks Like That: Deep Learning for Interpretable Image Recognition*](https://arxiv.org/pdf/1806.10574.pdf)

* **Links from Lecture 1**:
  * [Tay (bot)](https://en.wikipedia.org/wiki/Tay_(bot))
  * [New York Regulator Probes UnitedHealth Algorithm for Racial Bias](https://www.wsj.com/articles/new-york-regulator-probes-unitedhealth-algorithm-for-racial-bias-11572087601)
  * [When a Computer Program Keeps You in Jail](https://www.nytimes.com/2017/06/13/opinion/how-computers-are-harming-criminal-justice.html)
  * [When an Algorithm Helps Send You to Prison](https://www.nytimes.com/2017/10/26/opinion/algorithm-compas-sentencing-bias.html)

***

## Lecture 2: Post-hoc Explanation

![A decision tree surrogate model forms a flow chart of a more complex monotonic GBM](/img/lecture_2.png)
<sub><sup>**Source:** [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)</sup></sub>

### Lecture 2 Class Materials

* [Lecture Notes](tex/lecture_2.pdf) 
* [Lecture Video](https://www.youtube.com/watch?v=29wcG5wlRDw)
* Software Example: [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb) 
### Lecture 2 Suggested Software

* Python:
  * [`allennlp`](https://github.com/allenai/allennlp)
  * [`alibi`](https://github.com/SeldonIO/alibi)
  * [`interpret`](https://github.com/interpretml/interpret)
  * [`shap`](https://github.com/slundberg/shap)
  * [`tf-explain`](https://github.com/sicara/tf-explain)
* R:
  * [`DALEX`](https://cran.r-project.org/web/packages/DALEX/index.html)
  * [`iml`](https://cran.r-project.org/web/packages/iml/index.html)
  
### Lecture 2 Suggested Reading

* **Introduction and Background**: 
  * [*Proposed Guidelines for the Responsible Use of Explainable Machine Learning*](https://arxiv.org/pdf/1906.03533.pdf)
  * [**Responsible Artificial Intelligence**](https://www.springer.com/gp/book/9783030303709) - Chapter 4
  
* **Post-hoc Explanation Techniques**:
  * **Interpretable Machine Learning** - [Chapter 5](https://christophm.github.io/interpretable-ml-book/agnostic.html) and [Chapter 6](https://christophm.github.io/interpretable-ml-book/example-based.html)
  * [*Towards Better Understanding of Gradient-based Attribution Methods for Deep Neural Networks*](https://arxiv.org/pdf/1711.06104.pdf)

* **Links from Lecture 2**:
  * [On the Art and Science of Explainable Machine Learning](https://arxiv.org/pdf/1810.02909.pdf)
  * [Access Denied: Faulty Automated Background Checks Freeze Out Renters](https://themarkup.org/locked-out/2020/05/28/access-denied-faulty-automated-background-checks-freeze-out-renters)
  * [ML Attack Cheatsheet](https://github.com/jphall663/secure_ML_ideas/blob/master/img/cheatsheet.png)
  * [Debugging Machine Learning Via Model Assertions](https://cs.stanford.edu/~matei/papers/2019/debugml_model_assertions.pdf)
  * [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
  * [Gender Shades](http://gendershades.org/)
  * [Explainable Neural Networks based on Additive Index Models](https://arxiv.org/pdf/1806.01933.pdf)

***

## Lecture 3: Discrimination Testing and Remediation

![Two hundred neural networks from a random grid search trained on the UCI Credit Card Default dataset](/img/lecture_3.png)
<sub><sup>**Source:** [Lecture 3 Notes](tex/lecture_3.pdf)</sup></sub>

### Lecture 3 Class Materials

* [Lecture Notes](tex/lecture_3.pdf) 
* [Lecture Video](https://youtu.be/9kn1_E88wDA)
* Software Example: [Testing a Constrained Model for Discrimination and Remediating Discovered Discrimination](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_3.ipynb) 

### Lecture 3 Suggested Software

Python:

  * [`aequitas`](https://github.com/dssg/aequitas)
  * [`AIF360`](https://github.com/IBM/AIF360)
  * [`Themis`](https://github.com/LASER-UMASS/Themis)
  
### Lecture 3 Suggested Reading

* **Introduction and Background**:

  * **Fairness and Machine Learning** - [Introduction](https://fairmlbook.org/introduction.html)
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

## Lecture 4: 

![A cheatsheet for ML attacks](img/cheatsheet_blue.png)
<sub><sup>**Source:** [Secure Machine Learning Ideas](https://github.com/jphall663/secure_ML_ideas)</sup></sub>

### Lecture 4 Class Materials

* [Lecture Notes](tex/lecture_4.pdf)
* [Lecture Video](https://youtu.be/sdZ1udX6d6c)
* Software Example: [Attacking a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_4.ipynb) 

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
  * [*A Plea for Simplicity*](https://www.schneier.com/essays/archives/1999/11/a_plea_for_simplicit.html)
  * [*Privacy Risks of Explaining Machine Learning Models*](https://arxiv.org/pdf/1907.00164.pdf)

***   

## Lecture 5: Machine Learning Model Debugging

![Residuals for an important feature betray a serious problem in a machine learning model.](img/lecture_5.png)
<sub><sup>**Source:** [Real-World Strategies for Model Debugging](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)</sup></sub>

### Lecture 5 Class Materials

* [Lecture Notes](tex/lecture_5.pdf)
* [Lecture Video]()
* Software Example: [Debugging a Machine Learning Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_5.ipynb) 

### Lecture 5 Suggested Software

* [`DALEX`](https://cran.r-project.org/web/packages/DALEX/index.html) (R)
* [`SALib`](https://github.com/SALib/SALib) (Python)
* [`themis-ml`](https://github.com/cosmicBboy/themis-ml) (Python)
* [`What-if Tool`](https://pair-code.github.io/what-if-tool/index.html#about) (Browser)
  
  
### Lecture 5 Suggested Reading

* **Introduction and Background**:

  * [Debugging Machine Learning Models](https://debug-ml-iclr2019.github.io/)
  * [*Why you should care about debugging machine learning models*](https://www.oreilly.com/radar/why-you-should-care-about-debugging-machine-learning-models/)
  * [*Real-World Strategies for Model Debugging*](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)

* **Links from Lecture 5**:

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
