---
title: "Module 1 - Introduction"
classes: wide
---

## What Is Machine Learning?
* With the deluge of data, we need to find ways to discover what is in the data. ML is a set of algorithms/methods that help us learn and recognize the hidden patterns in data. ML is not a new topic. In fact, learning from data has been explored and used by many disciplines such as Statistics, Signal Processing, Control Theory, etc. What makes ML special is to provide a common formalism to the problems and algorithms. With the help of ML techniques, one can predict future data, or perform other kinds of decision-making under uncertainty.
* There are different types of ML. Sometimes models and methods are used interchangeably. However, a model is not a (learning) method/algorithm.
* Two common types of categorizing ML methods:
   - 1. Supervised Learning
   - 2. Unsupervised Learning
* Two common types of categorizing ML models:
   - 1. Discriminative
   - 2. Generative 

### Supervised Learning
* In supervised methods, we are given a set of \\(N\\) input-output pairs \\(\mathcal{D}=\\) \\(\\{(\mathbf{x_i}, y_i)\\}_{i=1}^N\\), and the goal is to learn a map from inputs, \\(\mathbf{x_i}\\)'s to outputs, \\(y_i\\)'s. Input variables have different names like **features**, **attributes**, or **covariates**. These input variables are typically a \\(p\\)-dimensional vector, denoting for example heights and weights of different persons (in this case, \\(p=2\\). That is, \\(\mathbf{x_i}\\) is a 2-dimensional real vector corresponding to the \\(i^{th}\\) person. However, input variables can be a very complex structured object, such as an image, a speech signal, a sentence, an email message, a time series, a graph, etc. On the other hand, output variables known as **response variable** or **labels** can be anything, but most methods assume that \\(y_i\\)'s are categorical or nominal variables from some finite set, i.e., \\(y_i\\) \\(\in\\) \\(\\{1,2,\dots,C\\}\\) in a classification problem, for example.

#### Supervised Problems Come in Two Flavors:
  1. **Regression:** In regression problems, the output variables are continuous, i.e., \\(y_i \in \mathbb{R}\\) or \\(y_i \in \mathbb{C}\\) for \\(i=1,2,\dots, N\\).
  2. **Classification:** In classification problems, the output variables are discrete, and they belong to a finite set (i.e., a set with a finite number of elements). That is, \\(y_i\\) \\(\in\\) \\(\\{1,2,\dots,C\\}\\) for \\(i=1,2,\dots, N\\).

      - **Face Detection** (regression example): The input, \\(\mathbf{x}\\) is an image, where \\(p\\) is the number of pixels in the image. The output, \\(y_i\\) is the location of faces in the figure (a real value).

         <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/38d8dfc0-7825-49f7-9993-09db19733f41">
               <br>
            <em>(a) Input image (Murphy family, photo taken 5 August 2010). (b) The output of the classifier, which detected 5 faces at different 
                poses. Classification example: Hand-written digit recognition. [K. Murphy, 2012.]</em>
         </p>

      - **Digit Recognition** (classification example): The input, \\(\mathbf{x}\\) is an image, where \\(p\\) is the number of pixels in the image. The output, \\(y_i\\) is one of the numbers in the set \\(\{0,1,2,\dots,9\}\\) (a discrete value).

         <p align="center">
            <img width="400" alt="Screenshot 2023-07-10 at 9 26 53 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/44375613-fbb2-4f22-a502-fbed168e471a">
            <br>
               <em>MNIST dataset. [http://yann.lecun.com/exdb/mnist/]</em>
         </p>

### Unsupervised Learning
* In unsupervised methods, we are only given input data without any labels. Here the goal is to discover any interesting or structure in the data (knowledge discovery). For example, discovering groups of similar examples within the data, where is called clustering. Another example is the density estimation problem, in which the goal is to estimate the distribution of data within the input space.

   - **Clustering** (image segmentation): Clustering or grouping similar pixels in an image.

      <p align="center">
        <img width="600" alt="Screenshot 2023-07-10 at 9 57 45 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/fb9ccc46-0a7b-4eb2-9396-a34642d1ff10">
        <br>
            <em>Application of the K-means clustering algorithm to image segmentation. [C. Bishop, 2006]</em>
      </p>
      
### Discriminative and Generative
* A discriminative model focuses on predicting labels of the input data, while a generative model explains how the data was generated. In other words, a discriminative model learns the boundary curve to distinguish data from each other. In the probabilistic language, it learns a conditional probability distribution given by \\(\mathbb{P}(Y\|\mathbf{X})\\). Please note that \\(Y\\) and \\(\mathrm{X}\\) are written as random (uppercase) quantities; however, we understand that these are events or realization vectors (such as \\(y_i\\)'s and \\(\mathbf{x_i}\\)'s). On the other hand, a generative model learns a joint probability distribution denoted by \\(\mathbb{P}(\mathbf{X}, Y)\\) (We will talk about our mathematical notations in the mathematics background modules).
  
   * Examples of discriminative models include Linear Regression, Logistic Regression, SVM, etc.
   * Examples of generative models include Linear Discriminant Analysis (LDA), Naive Bayes, Conditional GANs, Optical Flow Models (motion of objects in an image or a video sequence), etc.

### All Combinations Are Possible !!!

* There is a misconception that all generative models are unsupervised, or all discriminative models are supervised. This is obviously an incorrect statement.
   
   |              | Generative             | Discriminative |
   | -------------| ---------------------- | --------------------------------------------- |
   | **Supervised**   | Conditional GANs, Naive Bayes | SVM, Logistic Regression |
   | **Unsupervised** | LDA, Normalizing Flows | Optical Flow Models |

### Other Learning Methods
* In addition to the supervised and unsupervised learning methods, there are different learning approaches.
   *  **_Semi-Supervised Learning (SSL)._** There are some cases where we have only access to limited labels, but not enough to train a model in a supervised fashion. In this case, one needs to use both supervised and unsupervised techniques.
      * One approach to SSL is called _self-training_. The idea is to leverage the existing labels by training an initial model on a few labeled samples. This generates so-called Pseudo Labels. Next, we select more confident labels and construct a new dataset with the more confident pseudo-labeled and the limited labeled data, and train the initial model again for this new dataset. This hopefully improves the initial model. We then iteratively apply this procedure until the desired performance is met.

      <p align="center">
         <img src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/e0a94596-5de6-4f1f-8a7f-6b1891dccdb6" width="400" height="300">
         <br> 
               <em>Self-training approach for SSL.</em>
      </p>
      
   * **_Active Learning_.** Similar to semi-supervised learning, active learning provides another way to mitigate the issue of lack of enough labeled data. In particular, active learning starts with training an initial model using limited labeled data. It then tries to rank the unlabeled data using a method called _aqcuisition function_. Once the unlabeled data are ranked, those with higher ranks are typically labeled by a human, and then the model is again trained with both the small labeled dataset and the newly labeled data. Depending on what acquisition function is used, there are different types of active learning algorithms.

  <p align="center">
     <img width="400" alt="Screenshot 2023-07-12 at 9 55 34 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/18727e75-2efa-4199-a894-1d76e3d74667">
   <br>
  <em>Active Learning. [Burr Settles et al.].</em>
   </p>
     
   * **_Online Learning_.** Methods under this category try to learn from a stream or sequence of data instances arriving one by one at each time. So far, all discussed methods assume that the entire training data is available. Sometimes this is called _batch (off-line) learning_. The goal of online learning is to maximize accuracy/correctness for the sequence of predictions based on the history of received data and predictions made so far, and possibly additional available information. In general, online learning can be fully supervised, partially supervised (with partial/limited feedback), or totally unsupervised. Those online learning problems based on partial feedback are also called _multi-armed bandit_ problems. Online learning is a very useful learning technique as it overcomes the drawbacks of traditional batch learning in which the model can be updated dynamically by an online learner once new training data arrives. Moreover, online learning algorithms are often easy to understand, simple to implement, and often supported by solid theoretical analysis.
  
   * **_Reinforcement Learning (RL)_.** RL is another class of learning algorithms applied in scenarios where a decision should be made at each time and the information arrives sequentially from the environment. This scenario is similar to the case of online learning; however, there are no full/complete labels like supervised methods. Instead, all a learner receives feedback (known as reward) from the environment. The second class of online learning problems (multi-armed bandit problems) is a subset of the RL problem; although, in the online learning setup, the dynamic of the training environment is not explicitly modeled. The following figure is a standard schematic to illustrate an RL framework.
  
   <p align="center">
        <img width="400" alt="Screenshot 2023-07-12 at 9 55 34 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/3be81d4b-64fc-4e23-a8d3-fb5823fd609a">
      <br>
      <em>Standard Reinforcement Learning (RL) scenario.</em>
   </p>

* The following picture summarizes three important categories of ML approaches.

   <p align="center">
   <img width="400" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/3352531c-7ad3-4a13-bbac-8ef853bbb068">
   <br>
      <em>Three types of Machine Learning. [K. Murthy, 2022 & Yann LeCun at NIPS’16].</em>
   </p>

* In addition to the above learning methods, you may see names like _Transfer Learning_, _Meta Learning_, _Continous or Life-Long Learning_, _Curriculum Learning_, _Teacher-Student Learning (Distillation)_ in the literature. All these methods are different types of learning which may be used in supervised/unsupervised/RL fashion. We will look at some of these in the course.

### Datasets in ML
* As mentioned earlier, the availability of data is essential for all ML algorithms. In recent years, there are a large amount of data publicly available for training/evaluating/testing of ML algorithms. Here, we briefly review some of the common datasets used in ML literature.
   * Small dataset for ML:
      - 1. **Mall Customers.** This dataset contains information about people visiting the mall in a particular city, including age, annual income, and spending score. 
      - 2. **IRIS.** This dataset includes 150 sample with four features with information about the flower petal.
      - 3. **Boston Housing.** This dataset Contains information collected by the US Census Service about housing in the area of Boston Mass. The dataset is small in size with only 506 cases.
      - 4. **Fake News Detection.** This Dataset contains 7,796 data with with four features: news, title, news text, result.
      - 5. **Wine Quality.** The dataset contains different chemical information about the wine. Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). There are 1,599 red wine samples and 4,898 white wine samples.
      - 6. **SOCR data — Heights and Weights.** This dataset contains the height and weights of 25,000 different humans of 18 years of age.
      - 7. **Titanic.** The dataset contains information about name, age, sex, number of siblings aboard, and other information about 891 passengers in the training set and 418 passengers in the testing set.
      - 8. **Credit Card Fraud Detection.** The dataset contains transactions made by credit labeled as fraudulent or genuine. There are a total of 284,807 transactions, out of which only 492 are fraudulent (highly imbalanced dataset).
  
   * Data in image domain:
      - 1. **MNIST.** This dataset contains the gray-scale of \((28\times\)) images of digits from 0 to 9. It contains 60,000 training images and 10,000 testing images.
      - 2. **Fashion MNIST.** This dataset is similar to the original MNIST images, but the images are different objects, including 10 labels listed as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Angel boot.
      - 3. **CIFAR-10/100.** The CIFAR-10 dataset contains 60,000 colorful images of \((32\times32\)) pixels. They are labeled from 0-9. The CIFAR-100 is similar to the CIFAR-10 dataset, but the difference is that it has 100 classes instead of 10.
      - 4. **ImageNet.** This dataset contains 14,197,122 annotated images according to the WordNet hierarchy. Since 2010 the dataset is used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), a benchmark in image classification and object detection. The training data contains 1000 categories and 1.2 million images.
      - 5. **MS COCO.** This dataset is a huge database for object detection, segmentation and image captioning tasks. It has around 1.5 million labeled images.
      - 6. **Flickr 8k.** The Flickr 8k dataset contains 8,000 images and each image is labeled with 5 different captions. The dataset is used to build image captioning.
  
   * Data in Natural Language Processing (NLP):
      - 1. **Enron Email.** This Enron dataset contains around 0.5 million emails of over 150 users mostly from the management of Enron. The size of the data is around 432 MB.
      - 2. **Yelp.** It contains 1.2 million tips by 1.6 million users, over 1.2 million business attributes and photos for natural language processing tasks.
      - 3. **IMDB Reviews.** It contains over 50,000 movie reviews from Kaggle.
      - 4. **Amazon Reviews.** This dataset contains over 45 million Amazon reviews.
      - 5. **Rotten Tomatoes Reviews.** The dataset includes an archive of more than 480,000 critic reviews (fresh or rotten).
      - 6. **SMS Spam Collection in English.** This dataset includes 5,574 English SMS spam messages.
      - 7. **Twitter US Airline Sentiment.** The dataset consists of 55,000 Twitter data of major U.S. airline was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service").

   * Data in Speech:
      - 1. **LibriSpeech.** This dataset includes records for a large-scale (1000 hours) corpus of read English speech.
      - 2. **Multilingual LibriSpeech.** The dataset is a large multilingual corpus derived from LibriVox audiobooks.
      - 3. **HarperValleyBank.** The dataset includes simulated contact center calls to Harper Valley Bank in the Gridspace Mixer platform. The records have been labelled with human transcripts, timing information, emotion and dialog acts model outputs, etc.
      - 4. **Common Voice.** This dataset contains 7,335 validated hours of speech in 60 languages. Each entry in the dataset consists of a unique MP3 and corresponding text file.
      - 5. **TED-LIUM.** This is a dataset with 452 hours of audio from TED talks.
      - 6. **AudioMNIST.** The dataset consists of 30,000 audio samples of spoken digits (\((0-9\))) of 60 different speakers.
      - 6. **Google Speech Commands.** The dataset is consists of 65,000 one-second long utterances of 30 short words, by thousands of different people.
      - 7. **CHiME.** The CHiME-Home dataset is a collection of annotated domestic environment audio recordings.
      - 8. **Urban Sounds.** This dataset contains 1302 labeled sound recordings. Each recording is labeled with the start and end times of sound events from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.
