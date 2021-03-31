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
The paper is provided with code, which is written in Pytorch. The paper provides little guidance on steps to take to achieve the desired results and the information how to use their code and dataset is very sparse. This chapter first provides some insights in the code and which parts are needed to reproduce the results that we want to achieve. Afterwards practical steps to run the code are explained. The reproduced paper can be found here: https://arxiv.org/abs/2004.03459.  

### Explanation code and reproduction
**Order-preserving embedding methods**
The reproduced paper focusses on embedding-based approaches. An embedding can be seen as general and powerful approach that maps object to points in a high-dimensional space. This space is called the embedding space. In case of an order-preserving embedding semantic hierarchy is modelled by exploiting a structure on the quiescent space. The advantage over the distance-preserving methods is that anti-symmetric and transitive relations can be captures, without the need to rely of physical closeness between points. To deal with the large volume that each object occupies in the embedding space the paper uses Euclidean Entailment Cones (EC). This method uses more flexible convex cones. 
Another more general and flexible methods that is studied is one where embedding are defined by Hyperbolic Cones (HC). The advantage of HC over EC is that it does not have to deal with the capacity problem that EC has. This is due to the fact that the volume of a sphere grows exponentially in HC, compared to polynomial in EC. For this reason HC can visualize large trees. In the paper HC also shows better results than EC and for this reason we chose to focus on HC. 
 
https://arxiv.org/pdf/1511.06361.pdf 
https://arxiv.org/pdf/1804.01882.pdf 
**CCN-based methods**
Another possibility to preserve hierarchy in by incorporating it into the loss function of the model. These methods are less flexible, but can definitely improve performance compared to hierarchy-agnostic methods. In the paper 4 different CNN based methods are compared to a hierarchy agnostic baseline classifier and all 4 outperform the baseline. They even slightly outperform order-embeddings methods, although as mentioned before they are less flexible and do not allow for hierarchy retrieval. 
The baseline is Hierarchy-agnostic Baseline classifier (HAB) which is agnostic to any label hierarchy in the dataset. This model is a multi-label classifier and the multi-label soft-margin loss is minimized. The best performing classifier with hierarchy included in the loss function is the Marginalization Classifier (MC). The probability to belong to a class is not predicted per level, but only for the last level. 

### Practical steps and guidance on how to run the code


## Datasets
Levi
Met leuke plaatjes van beide structuren enzo. 


## Results
Both
