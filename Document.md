---
subtitle: |
    Academic Year 2020/21

    Barsellotti Luca, Bongiovanni Marco, Gholami Sina and Paolini Emilio
title: Multimedia Information And Computer Vision Project
---

<span style="font-variant:small-caps;">Index</span>

Introduction
============

The aim of this project is to build a Web Search Engine based on art
images, which works efficiently thanks to a Vantage Point Tree Index
approach. The list of images that can be retrieved are contained in the
“Art Dataset”, a dataset split into five classes (“Drawings”,
“Engraving”, “Iconography”, “Paintings” and “Sculpture”), and in a
“Distractor Dataset”, based on the Mirflickr25k dataset. To compare the
images, evaluate their similarity and build the index, we use a
Convolutional Neural Network. In particular, we use two models: one is
based on a Pre-Trained Network (“InceptionV3”) with weights computed on
“Imagenet” and the other one is a fine-tuned version of the same
Pre-Trained Network through the “Art Dataset”.

Dataset Analysis
----------------

### Training Set

First of all, we have analyzed the datasets. Both “Art Dataset” and
“Distractor Dataset” contained some corrupted images and so we have
removed them. At the end of the removing process, the training dataset
contained in the “Art Dataset” was composed by

Total: 7721\
Drawings: 1107\
Engraving: 757\
Iconography: 2077\
Painting: 2042\
Sculpture: 1738

![](media/image6.png){width="3.7014873140857394in"
height="2.7575546806649167in"}

### Test Set

The test set contained into the “Art Dataset” is composed by

Total: 856\
Drawings: 122\
Engraving: 84\
Iconography: 231\
Painting: 228\
Sculpture: 191

The distribution of the classes is the same of the “Training Set”.

Vantage Point Tree Implementation
=================================

### Introduction

The Vantage Point Tree is an Exact Similarity Searching Method based on
ball regions that recursively divide the given datasets. In particular,
at each step a vantage point *p* (also called pivot), belonging to the
actual dataset *X*, is chosen. After that, the median m of all the
distances from *p* of the points belonging to X is computed

$m = median(d\left( x,p \right))\ \ \ \ \forall x \in X$

and the dataset is split into two subsets according to:

$S_{1} = \left\{ x \in X - \left\{ p \right\}|d\left( x,\ p \right) \leq m \right\}$

$S_{2} = \left\{ x \in X - \left\{ p \right\}|d\left( x,\ p \right) \geq m \right\}$

At the end, the objects are stored into the leaves. The resulting
Vantage Point Tree is a balanced binary tree. This approach requires a
time cost approximately of $O(n\log\log\ n\ )$ during the building phase
and $O\left( \log\log\ n\  \right)$ in a 1-NN search.

### Build

The Vantage Point Tree has been built using the approach described in
the introduction, using as stopping condition the maximum bucket size
that can be contained by a leaf.

1.  **def** build(node, feature\_subset, leaf=False):  

2.      **if** leaf **is** True:  

3.          node.set\_leaf(feature\_subset)  

4.      **else**:  

5.          pivot = random.choice(feature\_subset)  

6.          distances = list()  

7.          **for** feature\_schema **in** feature\_subset:  

8.              distances.append(feature\_schema.distance\_from(pivot))  

9.          distances = np.array(distances)  

10.         median = np.median(distances)  

11.         subset1 = list()  

12.         subset2 = list()  

13.         **for** feature\_schema **in** feature\_subset:  

14.             **if** feature\_schema.distance\_from(pivot) &lt;= median:  

15.                 subset1.append(feature\_schema)  

16.             **else**:  

17.                 subset2.append(feature\_schema)  

18.         node.set\_median(median)  

19.         node.set\_pivot(pivot)  

20.         node.add\_left(Node())  

21.         node.add\_right(Node())  

22.         nodes += 2  

23.         **if** len(subset1) &lt;= bucket\_size:  

24.             build(node.get\_left(), subset1, leaf=True)  

25.         **else**:  

26.             build(node.get\_left(), subset1)  

27.         **if** len(subset2) &lt;= bucket\_size:  

28.             build(node.get\_right(), subset2, leaf=True)  

29.         **else**:  

30.             build(node.get\_right(), subset2)  

### Range Search

During the Range Search, at each step the distance between the query
object and the pivot is evaluated to decide if it is necessary to access
the children nodes. The condition to access the left node is

$d(q,p) - r \leq m$

instead, the condition to access the right node is

$d(q,p) + range \leq m$

If the node considered in the step is a leaf, then all the distances
from the objects contained in its bucket are evaluated to decide if they
have to be added to the result.

1.  **def** range\_search(node, query, range):  

2.      **if** node.is\_leaf() == True:  

3.          **for** object **in** node.get\_subset():  

4.              **if** query.distance\_from(object) &lt;= range:  

5.                  range\_search\_return\_list.append(object)  

6.          **return**  

7.      **if** query.distance\_from(node.get\_pivot()) &lt;= range:  

8.          range\_search\_return\_list.append(node.get\_pivot())  

9.      **if** query.distance\_from(node.get\_pivot()) - range &lt;= node.get\_median():  

10.         recursive\_range\_search(node.get\_left(), query, range)  

11.     **if** query.distance\_from(node.get\_pivot()) + range &gt;= node.get\_median():  

12.         recursive\_range\_search(node.get\_right(), query, range)  

13.     **return**  

### K-Nearest Neighbors Search

In the k-NN Search a Priority Queue is used to maintain the retrieved
objects ordered by the distance. This Priority Queue is initialized with
the first objects encountered while traversing the Vantage Point Tree.
At each step, if the node is an internal node then the distance between
the query and its pivot is computed. If this distance is lower than
$d_{\text{MAX}}$, the current highest distance in the Priority Queue,
the pivot is added to the queue. Hence, if

$d(q,p) - d_{\text{MAX}} \leq m$

then the k-NN Search is called on the left node and if

$d(q,p) + d_{\text{MAX}} \geq m$

then the k-NN Search is called on the right node. Instead, if the node
is a leaf, the distances from all the objects are computed to evaluate
if they have to be added in the Priority Queue.

1.  **def** knn(node, query, k):  

2.      **if** node.is\_leaf() == True:  

3.          **for** object **in** node.get\_subset():  

4.              distance = query.distance\_from(object)  

5.              **if** **not** nn.is\_full():  

6.                  nn.push(priority=distance, item=object)  

7.                  d\_max = distance **if** d\_max &lt; distance **else** d\_max  

8.              **elif** abs(distance) &lt; abs(d\_max):  

9.                  nn.pop()  

10.                 nn.push(priority=distance, item=object)  

11.                 tmp = nn.data\[0\]\[0\]  

12.                 d\_max = abs(tmp)  

13.         **return**  

14.     distance = query.distance\_from(node.get\_pivot())  

15.     **if** **not** nn.is\_full():  

16.         nn.push(priority=distance, item=node.get\_pivot())  

17.         d\_max = distance **if** d\_max &lt; distance **else** d\_max  

18.     **elif** abs(distance) &lt; abs(d\_max):  

19.         nn.pop()  

20.         nn.push(priority=distance, item=node.get\_pivot())  

21.         tmp = nn.data\[0\]\[0\]  

22.         d\_max = abs(tmp)  

23.     **if** distance - d\_max &lt;= node.get\_median():  

24.         knn(node.get\_left(), query, k)  

25.     **if** distance + d\_max &gt;= node.get\_median():  

26.         knn(node.get\_right(), query, k)  

27.     **return**  

Time Performance
----------------

### Range Search with vptree

### Range Search without index

### K-NN with vptree

### k-nn not optimized with vptree

### k-nn without index

Feature Extraction with a Pre-Trained Network
=============================================

### Structure of inception v3

Inception v3 is a Convolutional Neural Network model based on the paper
“Rethinking the Inception Architecture for Computer Vision” presented by
Szegedy et al. that showed an 78.1% accuracy on the ImageNet dataset.
Its structure is composed by 42 layers split in different modules.

![](media/image2.png){width="6.6930555555555555in"
height="2.970138888888889in"}

The main module and the core idea of the Inception model that made it
one of the most used pre-trained models is the Inception Layer:

![](media/image1.png){width="5.541666666666667in"
height="2.9895833333333335in"}

This layer is used in the Neural Network to select the filter size that
will be relevant to learn the required information of the next layer, to
learn the best weights and so automatically select the more useful
features. This job is performed by the parallel Convolution Filters and
the 1x1 Convolution Filters are used to perform dimensionality
reduction.

### Performance

Feature Extraction with a Fine-Tuned Network
============================================

### Structure

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
Layer (type) Output Shape Param \#\
=================================================================\
inception\_v3 (Functional) (None, 5, 5, 2048) 21802784\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
gap (GlobalAveragePooling2D) (None, 2048) 0\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
dense (Dense) (None, 512) 1049088\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
dropout (Dropout) (None, 512) 0\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
last\_relu (Dense) (None, 512) 262656\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
last\_dropout (Dropout) (None, 512) 0\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\
classifier\_hidden (Dense) (None, 5) 2565\
=================================================================\
Total params: 23,117,093\
Trainable params: 23,082,661\
Non-trainable params: 34,432\
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

### Training (early stop, class weights, data AUGMENTATION...)

### Performance until gap

### Performance until dense

Performance comparison between Networks
=======================================

### Plots of performance

User Interface for Web Search Engine
====================================

### Architecture of the web search engine

### Screenshots

![](media/image5.png){width="6.6930555555555555in"
height="3.5909722222222222in"}

![](media/image4.png){width="6.6930555555555555in"
height="3.5909722222222222in"}

![](media/image3.png){width="6.6930555555555555in"
height="3.5909722222222222in"}
