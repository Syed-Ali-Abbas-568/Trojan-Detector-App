# Trojan Detection Web App

## Overview

The Trojan Detection Web App is a Streamlit-based web application for predicting and analyzing network flows to identify potential trojan activity. It utilizes a Gradient Boosting Classifier trained on network flow data to make predictions.

## Dataset Used:
The trojan detection dataset from kaggle was used for training and testing the model.
[Dataset link](https://www.kaggle.com/datasets/subhajournal/trojan-detection)
## Features

- Trojan prediction for network flows.
- Display of individual flow diagnoses and overall statistics.
- Visualization of the confusion matrix, ROC curve, and other metrics.

## Installation

1. Clone the repository:

```bash
   git clone https://github.com/yourusername/trojan-detection-web-app.git
```

2. Install the required dependencies

```bash
  pip install -r requirements.txts
```

3. Run the web app:
```bash
  streamlit run Trojan-web-app.py
```


## Usage

1.  Upload a CSV file containing network flow data.
2.  Optionally, upload a key file for evaluation.
3.  Explore the trojan predictions, confusion matrix, and other visualizations.



