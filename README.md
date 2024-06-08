# EMG-based-Gesture-Classification-using-Graphs-and-GNN

The project introduces a novel approach to EMG-based upper-limb gesture recognition systems, leveraging advancements in deep learning and graph neural networks. Traditional methodologies often fail to capture relational information within multi-channel EMG sensor networks, limiting model performance, generalizability, and interpretability. To address these challenges, the project presents meticulously crafted graph structures to encapsulate spatial and temporal relationships of EMG sensors and signals. By employing Graph Neural Network (GNN)-based classification models, the project achieves state-of-the-art performance across various gesture recognition tasks while providing interpretable insights into muscular activation patterns. The effectiveness of the proposed approach in maintaining high accuracy even with reduced sensor configurations suggests its potential for integration into AI-powered rehabilitation strategies.



## Table of Contents

- [Installation](#installation)
- [Dataset](#Dataset)
- [Structure](#Structure)
- [License](#license)

## Installation
Include the following dependencies 
1. Requirements for Python (we use Python 3.10.14)
2. We use Jupyter Notebook and other Integrated Development Environments (IDEs) commonly used for Python development.
3. To install the project dependencies, run:
```bash
pip install -r Others/requirements.txt
```

## Dataset

In our experiments, we utilize the following public EMG-gesture datasets. See the following link or each dataset folder(Readme.md) for details.
- [EMG Data for Gestures](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
- [CapgMyo Type A and Type B](https://www.mdpi.com/1424-8220/17/3/458)
- [Ninparo Database 5](https://ninapro.hevs.ch/instructions/DB5.html)
- [BandMyo Dataset](https://github.com/Agire/BandMyo-Dataset)

## Structure

In the training process of each dataset type, the model training procedure comprises three distinct file types:
- utils.py: This file encompasses data preprocessing and engineering tasks conducted on raw datasets. It includes windowing techniques to segment data and allocate it into node features for each datum.
- config.py: Within this file lies the structural configuration for data processing. It is responsible for constructing input graphical structures that mimic the geometrical formulation of each sEMG electrode.
- model.py: This file houses the implementation of the graph neural network model, specifically a graph convolutional network (GCN). The model utilizes the graphical structures and node features to train on different gestures using specific EMG gesture datasets.
- main.ipynb: This Jupyter Notebook serves as the main script for the training process. It showcases the pretrained models and provides visualizations of their outputs, thereby indicating their performance on each dataset.
- Readme.md: It contains the basic information of each dataset type utilized.



