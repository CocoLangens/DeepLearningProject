# Blog 
*Authors: Levi Kieft (l.kieft@student.tudelft.nl), Coco Langens (c.a.langens@student.tudelft.nl)*
## Introduction
This blog aims to describe our efforts into reproducing the paper “Hierarchical Image Classification using Entailment Cone Embeddings”. This paper discusses methods to leverage information about the semantic hierarchy embedded in class labels to boost image classification performance.
Why would it be useful to include hierarchical information in classification problems? When humans get the task to classify an image, usually the first step is to determine the overarching membership of the image and then try to identify a more specific sublabel. For example when you see a picture taken in a forest, most people will identify trees, bushes and flowers, but the differences between a common hawthorn and a midland hawthorn might cause some troubles. The paper tries to use label hierarchy  to provide machine with a human understanding of concepts and related sub-concepts. 

Classification in Deep Learning usually uses an approach that assumes mutually exclusive and unstructured labels. However in real datasets, labels often have an underlying structure. Incorporating this structure in the model can be advantageous when training data is scarce.   
The paper studies different approaches to incorporate hierarchical information in CNN classifiers. First different baselines are compared where the hierarchy is incorporated in the loss function. Afterwards embedding based methods based on entailment cones are studied. 
We want to verify the claims that incorporating hierarchical information in loss function improves the performance of CNN based methods. For this we will reproduce the results in Table 2, where we will focus on the Hierarchy-agnostic baseline classifier (HAB), Marginalization Classifier (MC) and the order-preserving embedding method using Hyperbolic Cones (HC). We will first use the ETHEC dataset as used in the paper, and afterwards verify the results with  a new dataset that we will build ourself using a subset of Imagenet.   

## Understanding/running the code
### Original code
The paper is provided with code, which is written in Pytorch. The paper itself describes little about steps followed to achieve the desired results, and the authors provide little information on how to use their code and dataset. This chapter will first provide some insights in the code and which parts are needed to reproduce the results that we want to achieve. Afterwards practical steps to run the code are explained. The reproduced paper can be found here: https://arxiv.org/abs/2004.03459.   
INHOUD  
- Uitleg over de code en wat we precies willen reproduceren  
- Praktische stappen om de code te runnen  

## Datasets
Levi
Met leuke plaatjes van beide structuren enzo. 


## Results
Both
