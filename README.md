# Introduction
This blog aims to describe our efforts into reproducing the paper “Hierarchical Image Classification using Entailment Cone Embeddings”. This paper discusses methods to leverage information about the semantic hierarchy embedded in class labels to boost image classification performance.
Why would it be useful to include hierarchical information in classification problems? When humans get the task to classify an image, usually the first step is to determine the overarching membership of the image and then try to identify a more specific sublabel. Take for example a picture taken in a forest. Most people will identify trees, bushes and flowers, but the differences between a common hawthorn and a midland hawthorn probably causes some troubles for most people. The paper tries to use label hierarchy to guide machines in the same way as humans are guided by their understanding of concepts and related sub-concepts. Incorporating hierarchical structure can be advantageous when training data is scarce.
The paper studies different approaches to incorporate hierarchical information in CNN classifiers. First different baselines are compared where the hierarchy is incorporated in the loss function. Afterwards embedding based methods based on entailment cones are studied.
We want to verify the claims that incorporating hierarchical information in loss function improves the performance of CNN based classifiers. To achieve this we will reproduce the results in Table 2, where we will focus on the Hierarchy-agnostic baseline classifier (HAB), Marginalization Classifier (MC) and the order-preserving embedding method using Hyperbolic Cones (HC). We will first use the ETHEC dataset as used in the paper, and afterwards verify the results with  a new dataset that we will build ourselves using a subset of Imagenet.   


# Original Code & Reproduction
The paper is provided with code, which is written in Pytorch. The code can be found here: https://github.com/ankitdhall/learning_embeddings. The readme does not provide comprehensive information and little guidance on how to achieve the desired results is given. This chapter first gives deeper insight in the Hierarchy-Agnostic Baseline classifier (HAB), the Marginalization Classifier (MC) and the embedding based method using Hyperbolic Cones (HC). Afterwards practical details about how to run the code and reproduce the results of the paper are given.

## Order-preserving embedding methods
The reproduced paper focuses on embedding-based approaches for image classification. An embedding can be seen as a general and powerful approach that maps objects to points in a high-dimensional (embedding) space. In case of an order-preserving embedding the semantic hierarchy is modelled by exploiting a structure on the quiescent space. The advantage over the distance-preserving methods is that anti-symmetric and transitive relations can be captured, without the need to rely on physical closeness between points (1) 
To capture the semantic information the paper uses two functions: Euclidean Cones (EC) and Hierarchical Cones (HC). Euclidean functions are widely used, but face the problem that they are not able to deal with structures that are defined by non-Euclidean spaces.  Hyperbolic Cones are more general and flexible and have the advantage that they can deal with the capacity problem that EC have. The volume of a sphere grows exponentially in HC, where it grows polynomial in EC and for this reason HC can visualize larger trees (2). In the reproduced paper HC also shows better results in terms of performance and for this reason we chose to focus on reproducing the HC method.

## CCN-based methods
Another possibility to preserve hierarchy is by incorporating the structure into the loss function of a model. These methods are less flexible, but can definitely improve performance compared to hierarchy-agnostic methods. In the reproduced paper four different CNN based methods are compared to a hierarchy agnostic baseline classifier and all four outperform the baseline. They even slightly outperform order-embedding methods, although as mentioned before they are less flexible and do not allow for hierarchy retrieval.
The baseline is Hierarchy-agnostic Baseline classifier (HAB) which is agnostic to any label hierarchy in the dataset. This model is a multi-label classifier and the multi-label soft-margin loss is minimized. The best performing classifier with hierarchy included in the loss function is the Marginalization Classifier (MC). This method only provides a probability to belong to a class for the last level instead of a prediction per level.  
Practical guidance to perform the classification tasks
To reproduce the results in the paper the original splits of the ETHEC dataset (v0.1) are used as provided in the Adam1x branch of the github repository. The dataset can be downloaded here: https://www.research-collection.ethz.ch/handle/20.500.11850/365379. Prior to diving deeper into the code a brief note about the requirements. Running the requirements_3.6.txt does not work entirely on Windows systems and you will have to install torchvision and tensorboardX manually. 

## CCN-based methods
To run experiments with both baseline classifiers (HAB and MC) the file ethec_experiments.py is needed. The script uses a mix of absolute and relative paths and even the relative paths often do not direct to the correct folder. To be able to run the script the first step is to change the paths directing to the images and json files to your local folders.   286 – 316 
After adapting the paths the argument ‘--loss’ is used to determine which classifier is used.  The arguments do not exactly match the names of the classifiers as mentioned in the paper, but the relations are quite clear. For HAB use the argument multi_label, for PLC multi_level, MC last_level, M-PLC masked_loss and for HS hsoftmax. 
At this point you are all set to run the CNN based methods with the ETHEC dataset. To verify the results using your own dataset some additional changes are needed. 
The first step is to switch to the master branch. Working on the Adam1x branch is possible too, but you will have to make changes in the function def __getitem__(self, item) (line 2667) to make sure the paths to the images and json files are not based on the ETHEC dataset.
In the file db.py you have to replace the dictionary containing ETHEC labels with your own dictionary. The dictionary is located in the class ETHECLabelMap (line 17) . To be able to run MC another change has to be made in loss.py. In this file change torch.tensor() to torch.LongTensor() in line 85. 
We ran the experiments using HAB and MC with the best performing settings according to the paper. Both experiments had a batch size of 64, used Adam for 100 epochs and used the ResNet-50 variant. The learning rates varied and were set to 10-2 and 10-5 for respectively HAB and MC.     

## Order-preserving embedding methods
To run the order-preserving embedding method with Hierarchical Cones the file oe_h.py is used. Prior to changing the code you need to convert the json files to npy files. Secondly the paths in oe_h.py have the same structure as in ethec_experiments.py and we changed them in the same way as we explained before. 
To be able to use the npy files you have to allow pickle by adding allow_pickle=True to np.load(files.npy, allow_pickle=True) (line 2332-2339). Lastly labelmap is only defined in debug or merged mode. The labelmap is a class from data/db.py where all family labels are defined. Add labelmap = ETHECLabelMap() in line 2320 to be able to run in a different mode than debug or merged mode. 
We used Euclidean Cone Loss and at this point we ran into a KeyError related to the images in the ETHEC dataset. However the key that is missing is present in the dictionary, it is only located in a deeper layer. We did not manage to solve this problem, but tried to reach the authors to ask how to proceed. Unfortunately this meant that we did not manage to reproduce the results for HC.   

# Datasets
First the way the database used in the paper is set up is explained, how the files are called and stored. Then this will be related to the new dataset the method is applied to, examining how they are similar and different.
Data type: contained images
The original paper made use of the ETHEC dataset, the link to which is given above. Specifically they make use of v0.1. The downloadable file contains images in folders named with dates, within the folders there is no clear order in which the images are grouped. The images are of different butterfly specimens, 47.978 of them, 448x448 pixels in size.
Each image is of a single butterfly specimen placed on a white background, or for a few species a black background. Examples can be seen in XXXXXXXX.






Images ETHZ_ENT01_2018_01_16_118736 (left) and ETHZ_ENT01_2018_01_16_118776 (right) from folder 2018_01_16R

Examples given in the paper, figure 3

The images of the tinyimagenet dataset show a lot more variance, they are not all loose specimens on a clear background but rather images with varying positions, backgrounds and angles. The total size is 29.500 images, around half of the ETHEC dataset, and as we will get into later, very differently distributed. These images are 64x64 pixels. As opposed to the ETHEC dataset that only focusses on butterflies, the new dataset has a much broader scope. It has 200 classes, from cats to fire-hydrants. To be able to create a hierarchical dataset, only animals were included, which resulted in a total of only 59 classes. Example images of one of these classes can be seen in figure x. 
Instead of folders per date, the images are stored in folders named after their wordnet code (e.g. n02099601 for the golden retriever above), so with folder location and image name the wordnet structure can be retrieved, which we will use to create a hierarchy.
There is thus a large discrepancy in the type of images between the datasets. ETHEC has much larger images (49xamount of pixels) and about twice as many of them. Plus, the background is clear white or black and contains no extra information. Tinyimagenet  has smaller images with different species positions, camera angles and background information, plus there are less in total. However, as there are fewer classes, the average amount of images per class is 500 for tinyimagenet and only around 90 for ETHEC. 

## Contained information
The images themselves contain no hierarchical data, the name is not determined by species and the images have no watermark of any kind. The hierarchical data is stored in provided .json files, provided in the paper github. These already differentiate between the train, test and validation set in a 80-10-10 split. For each image, the following information is given: token, image_path, image_name, family, subfamily, genus, specific_epithet, subspecific_epithet, intraspecific_epithet, author, country, primary_division, dec_lat, dec_long and barcode. The latter half of this information is unused for this program, only up until the specific_epithet the data is relevant, and these shall be expanded upon. The token is a unique image ID used to call the data from the json file. The image_path is the path to the folder in which the image is stored. The image_name is the name given to each image, so that the program can import the file. Then the hierarchical data of the image is stored.
From the structure of the given json files, a new program was written that reads through all  the images in a given folder and uses the folder name to extract the wordnet hierarchy. For the 59 species the full wordnet hierarchy was printed and a tree created based on the similarities and deviations.  This tree was examined to create splits between layers such that each image has 4 layers, and there are 2 families. These layers were chosen such that each image has four layers, and the amount of splits between layers is approximately constant (that each layer is exponentially growing with the same base: 2.77^1=3, 2.77^2=8, 2.77^3=21 and 2.77^4=49 vs true 2, 9, 26, 59). Each image was then stored in a json file in the format as seen in figure. 
This can be divided into several json files, done with the same 80-10-10 split of the original database. For the unique token the name of the image was used, as each name is unique. The unused information from the original json files was not stored in the newly created one for the new database. 

## Data distribution: Long tailed versus uniform
As stated previously, the ETHEC dataset contains 47.978 images of butterfly specimens, each part of a given family, subfamily, genus and species. The distribution of these images per layer can be seen in figure xxxx. On all layers it has a non-uniform, tailed distribution, becoming increasingly steep at lower levels. The tinyimagenet dataset has an uneven distribution for the families, subfamilies and genera but the species are evenly distributed. For the distribution of the new dataset, see figure.

## Layers and visualisation
The paper made a visualisation of the distribution of their data with linked nodes. To visually compare the datasets, this was also done for the new dataset to see resemblance and differences. 

# Results
The results of our effort to reproduce the paper are summarized in table 1. The presented values are the micro-averaged F1 scores. 
F1 = (2*P*R)/(P + R) (with P=precision and R=recall). Micro-averaged means that the score for a metric is calculated by accumulating across all labels and then use these accumulated contributions to calculate the micro score. 

We did not manage to obtain equally promising results as stated in the paper and also the results with our own dataset and the ETHEC dataset differ from each other. Variability between both dataset can be caused by the different structure and the size of the datasets. 



Classifier
m-F1 
Overall
L1 - 
Family
L2 -  
Subfamily
L3 - 
Genus
L4 - Genus 
specific epithet
Paper
HAB
0.8147
0.9417
0.9446
0.8311
0.4578
Reproduction ETHEC
HAB










Reproduction Imagenet
HAB
0.6752
0.8766
0.7761
0.6063
0.4313
Paper
MC
0.9223
0.9887
0.9758
0.9273
0.7972
Reproduction ETHEC
MC
0.7914
0.9738
0.9485
0.7579
0.4854
Reproduction Imagenet
MC
0.8706
0.9681
0.9305
0.8305
0.7535


In Figure x & y the overall m-F1 scores for MC reproduction with Imagenet are plotted against the number of epochs for both test and training experiments. After 12 epochs the test m-F1 score starts to decrease overfitting seems to occur. 




In images … the overall m-F1 scores for our HAB reproduction with Imagenet are plotted against the number of epochs. Also the loss curves are plotted against the number of   epochs for both test and training experiments (figure .. & ..). After 25 epochs overfitting seems to occur. 













