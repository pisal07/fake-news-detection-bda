\# Multimodal Fake News Detection \& Real-Time Analytics Pipeline



An end-to-end Big Data Analytics pipeline developed in R to detect digital misinformation across heterogeneous data sources. Combines traditional NLP term-frequency matrices with engineered stylometric features, evaluates multiple architectures (Naive Bayes, Linear Support Vector Machines, Random Forest), and deploys a live Shiny decision-support application integrated with the Google Fact Check Tools API.



\---



\## System Architecture



The pipeline processes raw data through automated schema harmonization, stylometric feature extraction, model benchmarking, and artifact serving.



```

+--------------------------------------------------------------------+

| 1. Heterogeneous Ingestion (4 Sources: ISOT, WELFake, Scraped)    |

+--------------------------------------------------------------------+

&#x20;                                 |

&#x20;                                 v

+--------------------------------------------------------------------+

| 2. Harmonization, Reuters Leakage Removal \& Balanced Subsampling  |

+--------------------------------------------------------------------+

&#x20;                                 |

&#x20;                                 v

+--------------------------------------------------------------------+

| 3. Feature Engineering: TF Sparse DTM (98%) + Stylometrics         |

|    (Exclamation Density, Caps Ratios, Text Length Profiling)       |

+--------------------------------------------------------------------+

&#x20;                                 |

&#x20;                                 v

+--------------------------------------------------------------------+

| 4. Stratified Train/Test Split (80/20 Partition via caret)         |

+--------------------------------------------------------------------+

&#x20;           |                             |                          |

&#x20;           v                             v                          v

+-----------------------+   +------------------------+   +--------------------+

| Naive Bayes (Laplace) |   | Random Forest (Champion|   | Scaled Linear SVM  |

+-----------------------+   +------------------------+   +--------------------+

&#x20;           \\                             |                          /

&#x20;            -------------------> + ---------------- <----------------

&#x20;                                         |

&#x20;                                         v

+--------------------------------------------------------------------+

| 5. Evaluation, Visualizations (ROC, CM Heatmaps) \& Shiny Artifacts |

+--------------------------------------------------------------------+

```



\---



\## Benchmark \& Evaluation Results



Evaluated across a stratified 20% holdout test partition:



| Model Architecture | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score | AUC-ROC |

| :--- | :---: | :---: | :---: | :---: | :---: |

| \*\*Naive Bayes (Laplace = 1)\*\* | 76.70% | 78.78% | 87.03% | 0.8270 | 0.7562 |

| \*\*Linear SVM (Scaled, Cost = 1)\*\* | 88.50% | 89.95% | 92.34% | 0.9113 | 0.9181 |

| \*\*Random Forest (100 Trees) 🏆\*\* | \*\*93.30%\*\* | \*\*94.84%\*\* | \*\*94.69%\*\* | \*\*0.9476\*\* | \*\*0.9806\*\* |



\### Key Architectural Findings

\* \*\*Feature Scale Invariance:\*\* Random Forest achieved champion performance (93.3% accuracy, 0.9806 AUC) because decision trees split thresholds on isolated variables, naturally handling mixed continuous stylometrics and sparse binary vocabulary tokens ($0$ or $1$).

\* \*\*Mitigating Zero-Frequency Defect:\*\* Implementing Laplace smoothing on Naive Bayes stabilized posterior probability estimates, though the model's conditional independence assumption struggled with correlated sensationalist phrases.

\* \*\*Euclidean Normalization:\*\* Linear SVM performance reached 88.5% only after feature scaling (`scale = TRUE`), preventing continuous text lengths from dominating Euclidean distance calculations.



\---



\## Model Artifacts \& Visualizations



\### 1. Multi-Model ROC Comparison

!\[ROC Curves](assets/roc\_curves\_comparison.png)



\### 2. Feature Importance \& Predictive Signals

!\[Feature Importance](assets/feature\_importance\_rf.png)



\### 3. Confusion Matrix Analysis

| Naive Bayes | Random Forest (Champion) | Linear SVM |

| :---: | :---: | :---: |

| !\[NB](assets/confusion\_matrix\_nb.png) | !\[RF](assets/confusion\_matrix\_rf.png) | !\[SVM](assets/confusion\_matrix\_svm.png) |



\---



\## Installation \& Local Execution



\### Prerequisites

\* R (>= 4.2.0)

\* RStudio



\### 1. Clone the Repository

```bash

git clone \[https://github.com/](https://github.com/)<your-username>/fake-news-detection-bda.git

cd fake-news-detection-bda

```



\### 2. Install Dependencies

```r

install.packages(c(

&#x20; "tm", "SnowballC", "wordcloud", "caret", "e1071", 

&#x20; "randomForest", "ggplot2", "pROC", "reshape2", "RColorBrewer", "shiny", "httr", "jsonlite"

))

```



\### 3. Run Pipeline \& Launch Application

```r

\# Execute end-to-end model training, feature extraction, and benchmark export:

source("bdacode.R")



\# Launch the interactive Shiny analytical dashboard:

shiny::runApp("BDA\_Project.R")

```



\---



\## Tech Stack

\* \*\*Language:\*\* R

\* \*\*NLP \& Text Mining:\*\* `tm`, `SnowballC`

\* \*\*Machine Learning:\*\* `randomForest`, `e1071`, `caret`

\* \*\*Evaluation \& Visual Analytics:\*\* `pROC`, `ggplot2`, `reshape2`, `wordcloud`

\* \*\*Deployment \& Serving:\*\* R Shiny, Bootstrap UI, RESTful Google Fact Check API

