![A responsible machine learning workingflow](/img/rml_diagram_no_hilite.png)
##### A Responsible Machine Learning Workflow Diagram
<sub><sup>**Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3)</sup></sub>

## GWU_DNSC 6290: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, security and privacy in machine learning.

* Lecture 1: Interpretable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Discrimination Testing and Remediation
* Lecture 4: Machine Learning Security
* Lecture 5: Machine Learning Model Debugging
* Lecture 6: Responsible Machine Learning Best Practices

## Lecture 1: Interpretable Machine Learning Models

![Histogram, partial dependence, and ICE for a monotonic GBM and a credit card customer's most recent repayment status](/img/lecture_1.png)
<sub><sup>**Source:** [Building from Penalized GLM to Monotonic GBM](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1.ipynb)</sup></sub>

### Lecture 1 Class Materials

* [Syllabus](rml_syllabus_summer_2020.pdf)
* [Lecture Notes](tex/lecture_1.pdf)
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
  
## Using Class Software Resources

1. Install [Git](https://git-scm.com/downloads).

2. Clone this repository with the examples: `$ git clone https://github.com/jphall663/GWU_rml.git`

3. Install Anaconda Python 5.1.0 from the [Anaconda archives](https://repo.continuum.io/archive/) and *add it to your system path.*

4. Install `virtualenv`: `$ pip install virtualenv` 

5. Change directories into the cloned repository: `$ cd GWU_rml`

6. Create a Python 3.6 virtual environment: `$ virtualenv -p /path/to/anaconda3/bin/python3.6 env_rml`

7. Activate the virtual environment: `$ source env_rml/bin/activate`

8. Install the correct packages for the example notebooks: `$ pip install -r requirements.txt`

9. Start Jupyter: `$ jupyter notebook`
