## GWU_DNSC 6330: Course Outline

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
* [Software Example](https://colab.research.google.com/drive/1PHGCYRTAgiYbvC1fjd6xaLbg1nr2x0aH?usp=sharing)
* [Assignment 3](assignments/tex/assignment_3.pdf)
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignments/eval.ipynb?flush_cache=true)
  * [Full evaluations results]()
* Reading [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 4 and Chapter 10
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 3 Additional Materials](additional_materials/am3.md)

***   

## Lecture 4: Machine Learning Security

![A cheatsheet for ML attacks](img/Attack_Cheat_Sheet.png)
<sub><sup>**Source:** [Responsible Machine Learning](https://resources.oreilly.com/examples/0636920415947/blob/master/Attack_Cheat_Sheet.png)</sup></sub>

### Lecture 4 Class Materials

* [Lecture Notes](tex/lecture_4.pdf)
* Software Examples:
  * [Attacks for Red-teaming](https://colab.research.google.com/drive/1X1t1wqqVk8dlz1ubb0VBcLP_KFdP3HsE?usp=sharing)
  * [Data Poisoning](https://colab.research.google.com/drive/13hs11eJAEsX3ZAHA6oH1Lmi4I-dH7d1G?usp=sharing)
  * [Backdoor Attack](https://colab.research.google.com/drive/1QRCSW42L6wDs6ML9VgQu-xAbpdny4Mq1?usp=sharing)  
* [Assignment 4](assignments/tex/assignment_4.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://pages.dataiku.com/oreilly-responsible-ai), Chapter 5 and Chapter 11
* [Lecture 4 Additional Materials](additional_materials/am4.md)

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
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 5 Additional Materials](additional_materials/am5.md)

***   

## Lecture 6: Responsible Machine Learning Best Practices

![A responsible machine learning workingflow](/img/rml_diagram_no_hilite.png)

<sub><sup>A Responsible Machine Learning Workflow Diagram. **Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3).</sup></sub>

### Lecture 6 Class Materials

* [Lecture Notes](tex/lecture_6.pdf)
* [Assignment 6 (Final Assessment)](assignments/tex/assignment_6.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 1 and Chapter 12
* [Lecture 6 Additional Materials](additional_materials/am6.md)

***

## Lecture 7: Risk Mitigation Proposals for Language Models

![An illustration of retrieval augmented generation (RAG).](/img/rag.png)

<sub><sup>A diagram for retrieval augmented generation. **Source:** [Lecture 7 notes](tex/lecture_7.pdf).</sup></sub>

### Lecture 7 Class Materials

* [Lecture Notes](tex/lecture_7.pdf)
* [Software Example](https://drive.google.com/drive/folders/1eR4iNqP2bbQHtnQx7Sj8_VhS_AeWPzBo?usp=sharing)
* Reading: [_Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile_](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf), pgs. 1-12, 47-53 
* [Lecture 7 Additional Materials](additional_materials/am7.md)
