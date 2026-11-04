## GWU_DNSC 6330: Course Outline

Materials for a technical, nuts-and-bolts course about increasing transparency, fairness, robustness, and security in machine learning.

* Lecture 1: Explainable Machine Learning Models
* Lecture 2: Post-hoc Explanation
* Lecture 3: Bias Testing and Remediation
* Lecture 4: Machine Learning Security
* Lecture 5: Machine Learning Model Debugging
* Lecture 6: Foundation Models for Structured Data
* Lecture 7: Responsible Machine Learning Best Practices

Corrections or suggestions? Please file a [GitHub issue](https://github.com/jphall663/GWU_rml/issues/new).

***

## Preliminary Materials
 * [Syllabus](https://docs.google.com/document/d/1P9lpuRmTPtNJ0ZywBnF0QbkgWcBpd0NLeHkp2ydambI/edit?usp=sharing)
 * [Whiteboard Notation](https://docs.google.com/presentation/d/1Axf9dizaE3XvGRelBHfsnAlMUPFuMExQ2WNVwQBKMrw/edit?usp=sharing)

## Lecture 1: Explainable Machine Learning Models

![Histogram, partial dependence, and ICE for a monotonic GBM and a credit card customer's most recent repayment status](/img/ebm.png)
<sub><sup>**Source:** [Simple Explainable Boosting Machine Example](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_1_ebm_example.ipynb?flush_cache=true)</sup></sub>

### Lecture 1 Class Materials

* [Lecture Notes](tex/lecture_1.pdf)
* [Software Example](https://drive.google.com/file/d/1PnDSsNYRh1JNqZ3wyPgCA-KQxBop7y4L/view?usp=sharing)
* [Assignment 1](assignment/tex/assignment_1.pdf):
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignment/eval.ipynb?flush_cache=true)
  * [Full evaluations results](assignment/model_eval_2025_03_27_14_06_12.csv)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 2 (pp. 33 - 50) and Chapter 6 (pp. 189 - 217)
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 1 Additional Material](additional_material/am1.md)

***

## Lecture 2: Post-hoc Explanation

![A decision tree surrogate model forms a flow chart of a more complex monotonic GBM](/img/lecture_2.png)
<sub><sup>**Source:** [Global and Local Explanations of a Constrained Model](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/lecture_2.ipynb)</sup></sub>

### Lecture 2 Class Materials

* [Lecture Notes](tex/lecture_2.pdf)
* [Software Example](https://colab.research.google.com/drive/1X7hagDcdMEU_YrGxsAXrUZo1hWCVN--H?usp=sharing)
* [Assignment 2](assignment/tex/assignment_2.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 2 (pp. 50 - 80) and Chapter 6 (pp. 208 - 230)
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 2 Additional Material](additional_material/am2.md)
  
***

## Lecture 3: Bias Testing and Remediation

![Two hundred neural networks from a random grid search trained on the UCI Credit Card Default dataset](/img/lecture_3.png)
<sub><sup>**Source:** [Lecture 3 Notes](tex/lecture_3.pdf)</sup></sub>

### Lecture 3 Class Materials

* [Lecture Notes](tex/lecture_3.pdf)
* [Software Example](https://colab.research.google.com/drive/1PHGCYRTAgiYbvC1fjd6xaLbg1nr2x0aH?usp=sharing)
* [Assignment 3](assignment/tex/assignment_3.pdf)
  * [Model evaluation notebook](https://nbviewer.jupyter.org/github/jphall663/GWU_rml/blob/master/assignment/eval.ipynb?flush_cache=true)
  * [Full evaluations results](assignment/model_eval_2025_04_14_10_28_13.csv)
* Reading [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 4 and Chapter 10
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 3 Additional Material](additional_material/am3.md)

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
* [Assignment 4](assignment/tex/assignment_4.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 5 and Chapter 11
* [Lecture 4 Additional Material](additional_material/am4.md)

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
* [Assignment 5](assignment/tex/assignment_5.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 3 and Chapter 8
  * Check availablity through GWU Libraries access to O'Reilly Safari
* [Lecture 5 Additional Material](additional_material/am5.md)

***   

## Lecture 6: Foundation Models for Structured Data

Lecture 6 is a placeholder for the replacement material.

* [Placeholder lecture notes](tex/lecture_6.pdf)

***

## Lecture 7: Responsible Machine Learning Best Practices

![A responsible machine learning workflow](/img/rml_diagram_no_hilite.png)

<sub><sup>A Responsible Machine Learning Workflow Diagram. **Source:** [*Information*, 11(3) (March 2020)](https://www.mdpi.com/2078-2489/11/3).</sup></sub>

### Lecture 7 Class Materials

* [Lecture Notes](tex/lecture_7.pdf)
* [Assignment 6 (Final Assessment)](assignment/tex/assignment_6.pdf)
* Reading: [_Machine Learning for High-Risk Applications_](https://www.oreilly.com/library/view/machine-learning-for/9781098102425/), Chapter 1 and Chapter 12
* [Lecture 7 Additional Material](additional_material/am7.md)


## Example instructions for using class software in Google Colab

For a local checkout of this repository, use the uv setup in
[`python_environment.md`](python_environment.md).
The Colab steps below apply to copies of the hosted Google Drive notebooks.

* Create a folder in your GWU Google Drive `My Drive` called `DNSC_6330_Software`

* To run the lecture examples: 
  * Create a folder named `Lecture_01` inside the `DNSC_6330_Software`
  * Save a copy of the class `01_Explainable_AI_Models.ipynb` notebook into the `Lecture_01` folder by using the `File` ->  
`Save a Copy In Drive` menu options, or download the class notebook and upload to your folder
  * In cell 1 of the `01_Explainable_AI_Models.ipynb` update the path to the `Lecture_01` folder:
    * likely `%cd drive/My\ Drive/DNSC_6330_Software/Lecture_01/`
    * Use the `%cd` and `%ls` commands to find your folder if needed
    * Generally the `drive.mount('/content/drive/', force_remount=True)` command can only be used once in a Colab session, so use `%cd` and `%ls` commands in a different cell or restart your colab session if you see strange errors
  * Download the example data from: https://drive.google.com/drive/folders/1jYZvT1j5khFnJC5NSqNeGiCNoOeib9YK?usp=sharing (Click triangle beside `Data` at top -> Download, then unzip and upload into `DNSC_6330_Software` folder)
  * Download some necessary Python code from: https://drive.google.com/drive/folders/1BPXxGp0QAKRl1ZP6Vd1xKuCwitiLyuy6?usp=sharing (Click triangle beside `hrml_book` at top -> Download, then unzip and upload into `DNSC_6330_Software` folder)
  * Upload these folders into the `DNSC_6330_Software` folder

* To run the homework templates:
  * Create a folder called `assignment` in the `DNSC_6330_Software` folder
  * Download then upload the notebook `assign_1_template.ipynb` into the `assignment` folder
  * Create a folder called `data` in the `DNSC_6330_Software` folder
  * Download the assignment data from the class GitHub: https://github.com/jphall663/GWU_rml/tree/master/assignment/data
  * Unzip the data files into CSV files and upload them into the `data` folder
  * In the `assignment_1.ipynb` add a cell before cell 1 that installs `h2o` and `interpret`: `!pip install interpret h2o`
  * In the `assignment_1.ipynb` add a cell before cell 3 that connects the notebook to the data:

```
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
# may need to be updated to location on your drive
%cd drive/My\ Drive/DNSC_6330_Software/assignment/
%ls
```
* Whenever asked, allow Colab to connect to your Google drive
* Delete any `__pycache__` folders you see
* In the end the `DNSC_6330_Software` folder should look like: 

```
DNSC_6330_Software
├── assignment
│   ├── assign_1_template.ipynb
├── data
│   ├── hmda_test_preprocessed.csv
│   └── hmda_train_preprocessed.csv
├── Data
│   ├── backdoor_testing
│   │   ├── constrained_backdoor_output.csv
│   │   ├── constrained_output.csv
│   │   ├── overfit_backdoor_output.csv
│   │   ├── overfit_output.csv
│   │   └── test_data.csv
│   ├── credit_line_increase.csv
│   ├── data_dictionary.csv
│   └── synthetic_data.csv
├── hrml_book
│   ├── explain.py
│   └── partial_dep_ice.ipynb
└── Lecture_01
    └── 01_Explainable_AI_Models.ipynb

```
* You can use the following commands in a colab notebook to check your file structure:

```
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
# may need to be updated to location on your drive
%cd drive/My\ Drive/DNSC_6330_Software/
```

`!apt-get -y install tree`

`! tree`

* Add new assignment templates into the `assignment` folder
* Add new code examples to `Lecture_XX/XX_notebook_name.iypnb` folders
  * For Lecture 2 that would be `Lecture_02/02_Explainable_AI_Post_Hoc.ipynb`

## Licenses

The software is licensed under the [MIT License](SOFTWARE_LICENSE.txt). Course content is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](CONTENT_LICENSE.txt).
