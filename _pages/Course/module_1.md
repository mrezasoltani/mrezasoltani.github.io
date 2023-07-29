---
title: "Module 1 - Introduction"
classes: wide
---

Every day we hear many buzzwords and similar terms like Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), and Data Science (DS). Let me ask Google what these mean by typing "ml dl ai data science data engineering". Google returns results with newly added search results based on _Generative_ _AI_ (The date of this search in Google is July 21, 2023).

   <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/40958e51-1fb0-47f3-a83a-cb522321144a">
    <br>
            <em>Google search result for the query "ml dl ai data science data engineering".</em>
     </p>

Under the hood, Google is running my search question on its experimental conversational AI service, powered by LaMDA, called Bard to get back to me (later in this course we will talk about these technologies). This is what I get from Google, but how does BARD do this? The answer to this question is in this course. In a nutshell, Google uses a model trained on millions or billions of data to enable its conversational service for accurate answers. Starting by the end of 2022, and with the release of ChatGPT from OpneAI, generative AI and _Large Language Model (LLM)_ have shown a significant improvement in conversational AI with many applications. 

In this course, our goal is to understand ML concepts from scratch, so we can have enough background to grasp more advanced concepts like the above conversational AI models. Before diving into a more formal definition of ML concepts, it is worth seeing the relation between all the disciplines we used in the above example as a Venn diagram. 

   <p align="center">
               <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" 
    src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/aa5924ad-98a6-4fc7-8351-663a7397c338">
       <br>
            <em>The relation between AI, ML, DL, DL, and DS [https://iq.opengenus.org/].</em>
     </p>


## What Is Machine Learning?
* Machine learning is an application of AI that enables systems to learn and improve from experience without being explicitly programmed. With the deluge of data, we need to find ways to discover what is in the data. ML is a set of algorithms/methods that help us learn and recognize the hidden patterns in data. ML is not a new topic. In fact, learning from data has been explored and used by many disciplines such as Statistics, Signal Processing, Control Theory, etc. What makes ML special is to provide a common formalism to the problems and algorithms. With the help of ML techniques, one can predict future data, or perform other kinds of decision-making under uncertainty.
* For solving a problem using ML techniques, we always need to have some data related to the task/problem we want to solve. In fact, data is one of the most important pieces in all ML algorithms/methods. Before jumping to the ML concept, we need to introduce some terminologies regarding the data.
* Remember that data is a starting point for every ML technique. With the help of data, we can so-called _**Train**_ a model to do our desired goal. As a result, we need to have a set of _**Training Data**_ to teach a model. After training a model, we need to test our _**Trained**_ model. Hence, there should be another set of _**Test Data**_ to see the performance of our trained model. From this, we immediately see that these two datasets should be somehow related. but not the same. The former means that there should be something common between the training and test data. Later we will formalize this by saying that these two datasets should be independent samples from the same probability distribution. The second condition (not being the same) is necessary because we want to see how a trained model performs on unseen data; otherwise, it will be cheating if we evaluate the performance of a model on the same data it has been trained for. We will look at these things later in the course
 
* There are different types of ML. Sometimes models and methods are used interchangeably. However, a model is not a (learning) method/algorithm.
* Two common types of categorizing ML methods:
     1. Supervised Learning
     2. Unsupervised Learning
* Two common types of categorizing ML models:
     1. Discriminative
     2. Generative 

### Supervised Learning
* In supervised methods, we are given a set of \\(N\\) input-output pairs \\(\mathcal{D}=\\) \\(\\{(\mathbf{x_i}, y_i)\\}_{i=1}^N\\), and the goal is to learn a map from inputs, \\(\mathbf{x_i}\\)'s to outputs, \\(y_i\\)'s. Input variables have different names like **features**, **attributes**, or **covariates**. These input variables are typically a \\(p\\)-dimensional vector, denoting for example heights and weights of different persons (in this case, \\(p=2\\). That is, \\(\mathbf{x_i}\\) is a 2-dimensional real vector corresponding to the \\(i^{th}\\) person. However, input variables can be a very complex structured object, such as an image, a speech signal, a sentence, an email message, a time series, a graph, etc. On the other hand, output variables known as **response variable** or **labels** can be anything, but most methods assume that \\(y_i\\)'s are categorical or nominal variables from some finite set, i.e., \\(y_i\\) \\(\in\\) \\(\\{1,2,\dots, C\\}\\) in a classification problem, for example.

#### Supervised Problems Come in Two Flavors:
  1. **Regression:** In regression problems, the output variables are continuous, i.e., \\(y_i \in \mathbb{R}\\) or \\(y_i \in \mathbb{C}\\) for \\(i=1,2,\dots, N\\).

      - **Face Detection** (regression example): The input, \\(\mathbf{x}\\) is an image, where \\(p\\) is the number of pixels in the image. The output, \\(y_i\\) is the location of faces in the figure (a real value). Here, the goal is to find the location of faces in an input image by drawing a _bounding-box_ around each detected face.

         <p align="center">
            <img width="600" alt="Screenshot 2023-07-10 at 7 21 57 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/38d8dfc0-7825-49f7-9993-09db19733f41">
               <br>
            <em>(a) Input image (Murphy family, photo taken 5 August 2010). (b) The output of the classifier, which detected 5 faces at different 
                poses. Classification example: Hand-written digit recognition [K. Murphy, 2012.].</em>
         </p>

  2. **Classification:** In classification problems, the output variables are discrete, and they belong to a finite set (i.e., a set with a finite number of elements). That is, \\(y_i\\) \\(\in\\) \\(\\{1,2,\dots,C\\}\\) for \\(i=1,2,\dots, N\\).

      - **Digit Recognition** (classification example): The input, \\(\mathbf{x}\\) is an image, where \\(p\\) is the number of pixels in the image. The output, \\(y_i\\) is one of the numbers in the set \\(\{0,1,2,\dots,9\}\\) (a discrete value). Here, the goal is to classify an input image to one of \((10\)) possible classes. 

         <p align="center">
            <img width="400" alt="Screenshot 2023-07-10 at 9 26 53 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/44375613-fbb2-4f22-a502-fbed168e471a">
              <br>
            <em>MNIST dataset [http://yann.lecun.com/exdb/mnist/].</em>
         </p>

### Unsupervised Learning
* In unsupervised methods, we are only given input data without any labels. Here the goal is to discover any interesting structure in the data (knowledge discovery). For example, discovering groups of similar examples within the data, where is called clustering. Another example is the density estimation problem, in which the goal is to estimate the distribution of data within the input space. One more important example of unsupervised tasks is to decompose data/signals into simpler or more interpretable components. Two common tasks in this category are the _Factor Analysis_ methods and _Manifold Learning_ for linear and non-linear data, respectively (linear and non-linear data means that features/attributes underlying the data show a linear (non-linear) relationship or trend. We will talk more about this later).

   - **Clustering** (image segmentation): Clustering or grouping similar pixels in an image. Consider a task of grouping similar pixels of an image. For example, in the following figure, we are asked to divide pixels of the image into \\(K=2, 3, 10\\) groups. However, we have not been provided with any other information such as the label of the image, or some other information about the content of the image. Hence, our unsupervised algorithm (clustering algorithm) needs somehow to explore the similarity between pixels of the input image and group them in \\(K\\) groups/clusters. What similarity means is a broad concept. For instance, similar pixels mean those pixels whose values are close to each other. In this example, \\(K\\) clusters have been shown by different colors (e.g., for \\(K=2\\), there are only 2 types of pixels: blue or dark).

      <p align="center">
        <img width="600" alt="Screenshot 2023-07-10 at 9 57 45 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/fb9ccc46-0a7b-4eb2-9396-a34642d1ff10">
        <br>
            <em>Application of the K-means clustering algorithm to the image segmentation [C. Bishop, 2006]</em>
      </p>
      
   - **Principle Component Analysis (PCA)** (dimensionality reduction and visualization): PCA is used to decompose a multivariate dataset into components that explain a maximum amount of the variance. What does this mean? In many cases, we want to get some idea about the data we are dealing with. For example, we want to visualize data or understand which attributes of data have more information. Consider you are given a set of flower images, where each image is a 1-megapixel colorful image (an image with size (\\(667\times 500\times 3\\)). You can think of each image sample as living in a huge space (i.e., a space with dimension (\\(1,000,500=667\times 500\times 3\\)). Obviously, we cannot visualize this huge space. What if we can represent each image in a 3-d or even 2-d space such that only the most important attributes of data appear in this reduced-dimensionality space (please note that the meaning of _most important attributes_ needs to be defined carefully, but for now think about some attributes helping us to distinguish flower images by looking their representation in the new 2/3-d space). This is exactly what PCA can do for us; by projecting an image to a 2/3-d space such that only those 2/3 components explain the most important information about flower images. As we can see, only the first two components from the PCA can show three clear classes of the input images which originally have more than 1 M dimension.
 
     <p align="center">
        <img width="1500" alt="Screenshot 2023-07-10 at 9 57 45 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/a88a7265-fe1e-4629-82be-546a49245676">
        <br>
            <em>Application of the PCA for visualization. Flower images from [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/].</em>
      </p>
     
### Discriminative and Generative
* A discriminative model focuses on predicting labels of the input data, while a generative model explains how the data was generated. In other words, a discriminative model learns the boundary curve to distinguish data from each other. In the probabilistic language, it learns a conditional probability distribution given by \\(P(Y\|\mathbf{X})\\). Please note that \\(Y\\) and \\(\mathbf{X}\\) are written as random (uppercase) quantities; however, we understand that these are events or realization vectors (such as \\(y_i\\)'s and \\(\mathbf{x_i}\\)'s). On the other hand, a generative model learns a joint probability distribution denoted by \\(P(\mathbf{X}, Y)\\) (We will talk about our mathematical notations in the mathematics background modules).
  
   * Examples of discriminative models include Linear Regression, Logistic Regression, SVM, etc.
   * Examples of generative models include Linear Discriminant Analysis (LDA), Naive Bayes, Conditional GANs, Optical Flow Models (motion of objects in an image or a video sequence), etc.

### All Combinations !!!

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
         <img src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/e0a94596-5de6-4f1f-8a7f-6b1891dccdb6" width="500" height="350">
         <br> 
               <em>Self-training approach for SSL.</em>
      </p>
      
   * **_Active Learning_.** Similar to semi-supervised learning, active learning provides another way to mitigate the issue of lack of enough labeled data. In particular, active learning starts with training an initial model using limited labeled data. It then tries to rank the unlabeled data using a method called _aqcuisition function_. Once the unlabeled data are ranked, those with higher ranks are typically labeled by a human, and then the model is again trained with both the small labeled dataset and the newly labeled data. Depending on what acquisition function is used, there are different types of active learning algorithms.

  <p align="center">
     <img width="400" alt="Screenshot 2023-07-12 at 9 55 34 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/18727e75-2efa-4199-a894-1d76e3d74667">
   <br>
  <em>Active Learning [Burr Settles et al.].</em>
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
   <img width="600" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/3352531c-7ad3-4a13-bbac-8ef853bbb068">
   <br>
      <em>Three types of Machine Learning [K. Murthy, 2022 & Yann LeCun at NIPS’16].</em>
   </p>

* In addition to the above learning methods, you may see names like _Transfer Learning_, _Meta Learning_, _Continous or Life-Long Learning_, _Curriculum Learning_, _Teacher-Student Learning (Distillation)_ in the literature. All these methods are different types of learning which may be used in supervised/unsupervised/RL fashion. We will look at some of these in the course.

### Datasets in ML
* As mentioned earlier, the availability of data is essential for all ML algorithms. In recent years, there are a large amount of data publicly available for training/evaluating/testing of ML algorithms. Here, we briefly review some of the common datasets used in ML literature.
   * Small dataset for ML:
     1. **Mall Customers.** This dataset contains information about people visiting the mall in a particular city, including age, annual income, and spending score ([link](https://www.kaggle.com/shwetabh123/mall-customers)). 
     2. **IRIS.** This dataset includes 150 samples with four features with information about the flower petal ([link](https://archive.ics.uci.edu/ml/datasets/Iris)).
     3. **Boston Housing.** This dataset Contains information collected by the US Census Service about housing in the area of Boston Mass. The dataset is small in size with only 506 cases ([link](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)).
     4. **Fake News Detection.** This Dataset contains 7,796 data with four features: news, title, news text, result ([link](https://www.kaggle.com/c/fake-news/data)).
     5. **Wine Quality.** The dataset contains different chemical information about the wine. Each expert graded the wine quality between 0 (very bad) and 10 (very excellent). There are 1,599 red wine samples and 4,898 white wine samples ([link](https://archive.ics.uci.edu/ml/datasets/wine+quality)).
     6. **SOCR Data — Heights and Weights.** This dataset contains the height and weights of 25,000 different humans of 18 years of age ([link](http://wiki.stat.ucla.edu/socr/index.php/SOCR_Data_Dinov_020108_HeightsWeights)).
     7. **Titanic.** The dataset contains information about name, age, sex, number of siblings aboard, and other information about 891 passengers in the training set and 418 passengers in the testing set ([link](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html)).
     8. **Credit Card Fraud Detection.** The dataset contains transactions made by credit labeled as fraudulent or genuine. There are a total of 284,807 transactions, out of which only 492 are fraudulent (highly imbalanced dataset) ([link](https://www.kaggle.com/mlg-ulb/creditcardfraud)).
  
   * Data in image domain:
     1. **MNIST.** This dataset contains the gray-scale of \\(28\times 28\\) images of digits from 0 to 9. It contains 60,000 training images and 10,000 testing images ([link](http://yann.lecun.com/exdb/mnist/)).
     2. **Fashion MNIST.** This dataset is similar to the original MNIST images, but the images are different objects, including 10 labels listed as T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Angel boot ([link](https://www.kaggle.com/datasets/zalando-research/fashionmnist)).
     3. **CIFAR-10/100.** The CIFAR-10 dataset contains 60,000 colorful images of \\(32\times 32\\) pixels. They are labeled from 0-9. The CIFAR-100 is similar to the CIFAR-10 dataset, but the difference is that it has 100 classes instead of 10 ([link](https://www.cs.toronto.edu/~kriz/cifar.html)).
     4. **ImageNet.** This dataset contains 14,197,122 annotated images according to the WordNet hierarchy. Since 2010 the dataset is used in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), a benchmark in image classification and object detection. The training data contains 1000 categories and 1.2 million images ([link](http://image-net.org/)).
     5. **MS COCO.** This dataset is a huge database for object detection, segmentation, and image captioning tasks. It has around 1.5 million labeled images ([link](https://cocodataset.org/#home)).
     6. **Flickr 8k.** The Flickr 8k dataset contains 8,000 images and each image is labeled with 5 different captions. The dataset is used to build image captioning ([link](https://www.kaggle.com/datasets/adityajn105/flickr8k)).
     7. **Stanford Dogs.** This Dataset contains 20,580 images and 120 different dog breed categories ([link](http://vision.stanford.edu/aditya86/ImageNetDogs/)).
  
   * Data in Natural Language Processing (NLP):
     1. **Enron Email.** This Enron dataset contains around 0.5 million emails of over 150 users mostly from the management of Enron. The size of the data is around 432 MB ([link](https://www.cs.cmu.edu/~enron/)).
     2. **Yelp.** It contains 1.2 million tips by 1.6 million users, over 1.2 million business attributes, and photos for natural language processing tasks ([link](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)).
     3. **IMDB Reviews.** It contains over 50,000 movie reviews from Kaggle (25,000 reviews for training and 25,000 for the testing set) ([link](http://ai.stanford.edu/~amaas/data/sentiment/)).
     4. **Amazon Reviews.** This dataset contains over 45 million Amazon reviews ([link](https://snap.stanford.edu/data/web-Amazon.html)).
     5. **Rotten Tomatoes Reviews.** The dataset includes an archive of more than 480,000 critic reviews (fresh or rotten) ([link](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)).
     6. **SMS Spam Collection in English.** This dataset includes 5,574 English SMS spam messages ([link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)).
     7. **Twitter US Airline Sentiment.** The dataset consists of 55,000 Twitter data of major U.S. airlines was scraped from February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing negative reasons (such as "late flight" or "rude service") ([link](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)).

   * Data in Speech:
     1. **LibriSpeech.** This dataset includes records for a large-scale (1000 hours) corpus of read English speech ([link](http://www.openslr.org/12)).
     2. **Multilingual LibriSpeech.** The dataset is a large multilingual corpus derived from LibriVox audiobooks ([link](http://www.openslr.org/94)).
     3. **HarperValleyBank.** The dataset includes simulated contact center calls to Harper Valley Bank in the Gridspace Mixer platform. The records have been labeled with human transcripts, timing information, emotion, and dialog acts model outputs, etc ([link](https://github.com/cricketclub/gridspace-stanford-harper-valley)).
     4. **Common Voice.** This dataset contains 7,335 validated hours of speech in 60 languages. Each entry in the dataset consists of a unique MP3 and corresponding text file ([link](https://commonvoice.mozilla.org/en/datasets)).
     5. **TED-LIUM.** This is a dataset with 452 hours of audio from TED talks ([link](https://www.openslr.org/51)).
     6. **AudioMNIST.** The dataset consists of 30,000 audio samples of spoken digits (\((0-9\))) of 60 different speakers ([link](https://github.com/soerenab/AudioMNIST)).
     6. **Google Speech Commands.** The dataset consists of 65,000 one-second-long utterances of 30 short words, by thousands of different people ([link](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)).
     7. **CHiME.** The CHiME-Home dataset is a collection of annotated domestic environment audio recordings ([link](https://archive.org/details/chime-home)).
     8. **Urban Sounds.** This dataset contains 1302 labeled sound recordings. Each recording is labeled with the start and end times of sound events from 10 classes: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music ([link](https://urbansounddataset.weebly.com/)).

### Some Common Tasks in ML
* Before finishing this module, we are going to list some common tasks in different domains. Please note that this is just a tiny piece of possible tasks that can be accomplished with ML.
   * **Tasks in Computer Vision (CV).** The input data for this domain include images and videos. Of course, these types of datasets can be used in a multimodal task in which there might be other modalities like speech and text data.
  
     1. **Image Classification.** This task means identifying what class the object belongs to. Here \\(\mathbf{x}\\), the input variable is an image with objects we want to classify, and \\(y\\), the output variable is the label of different objects (i.e., cat=0, dog=1). This task is discriminative and supervised.
        
        <p align="center">
         <img width="640" alt="Screenshot 2023-07-16 at 11 29 17 AM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/0bb15098-6c60-44e0-8a20-316638896234">
             <br>
             <em>Classifying cat and dog images.</em>
        </p>
        
     2. **Object Detection.** Here \\(\mathbf{x}\\), the input variable is an image with objects we want to detect, and \\(\mathbf{y}\\) (please note we are using a bold letter to indicate that the output is a vector), the output variable is the location of objects in the image. Hence, we have a regression task. Please note that sometimes object recognition is used to indicate both the classification and detection of an object (In this case we have a hybrid task, including both classification and regression). In this sense, the output variable also includes the type (class) of the detected objects. This task is discriminative and supervised.
      
        <p align="center">
         <img width="706" alt="Screenshot 2023-07-16 at 11 26 45 AM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/a69ab827-d8dc-40fd-8cb3-d2baa5255d56">
             <br>
             <em>Object detection.</em>
        </p>
  
     3. **Image Segmentation.** This task is similar to object recognition but with very precise edge/contour detection. Image segmentation algorithms will segment and label all pixels that belong to an object. Here \\(\mathbf{x}\\), the input variable is an image with objects we want to segment, and \\(\mathbf{y}\\), the output variable is the image of the segmentation process (i.e., an image with labeled all the pixels that belong to every object). There are two types of segmentation: Instance Segmentation and Semantic Segmentation. An instance segmentation model outputs an image by separating every single object instance, while in semantic segmentation, the pixels of every instance of an object are labeled with the same class label. This task is considered a hybrid task, including both classification and regression, discriminative and supervised.

        <p align="center">
        <img width="770" alt="Screenshot 2023-07-16 at 12 31 19 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/0f42d816-9ea2-4a69-841b-4f53fd30055b">
             <br>
             <em>Image Segmentation [https://keras.io/examples/vision/image_classification_from_scratch/].</em>
        </p>
     
     4. **Image-to-Image** Here, the goal is to convert an input image, \\(\mathbf{x}\\) to an output image, \\(\mathbf{y}\\). The output image can be an improved version of the input image (deblurring an image, the super-resolution of the input image, image inpainting), or another image with some changes to the content of an image. This task is a supervised regression task which can be generative or discriminative.

        <p align="center">
        <img width="806" alt="Screenshot 2023-07-16 at 12 50 10 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/6eb093e7-b414-411d-9daf-4ae0285853c2">
             <br>
             <em>Image-to-Image task. Making girls show two fingers [https://osu-nlp-group.github.io/MagicBrush/].</em>
        </p>
     
     5. **Image Generation.** Here, the goal is to generate an output image, \\(\mathbf{y}\\) from typically a lower dimensional input vector, \\(\mathbf{x}\\). This task is a supervised/unsupervised generative task.
    
        <p align="center">
        <img width="628" alt="Screenshot 2023-07-16 at 1 07 42 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/1138fbce-0bbf-4503-8b0f-f425fbaeacb0">
             <br>
             <em>Image Generation.</em>
        </p>
 
     6. **Depth Estimation.** This is the task of measuring the distance of each pixel relative to the camera. Depth is extracted from either monocular (single) or stereo (multiple views of a scene) images. Here \\(\mathbf{x}\\), the input variable is an image, and \\(\mathbf{y}\\), the output variable is a heat-map image of the input image (i.e., an image which shows the depth of objects present as a heat-map).

        <p align="center">
        <img width="771" alt="Screenshot 2023-07-16 at 3 46 21 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/53898018-d0e7-4ccf-be30-88ccc3221eba">
             <br>
             <em>Image depth estimation [P. Hambarde et al., 2019].</em>
        </p>
        
    
   * **Tasks in Natural Language Processing (NLP).** The input data for this domain includes text coming from books, papers, chats, blogs, websites, transcriptions, emails, etc. Many of the following examples are from [Hugging Face](https://huggingface.co/).
     
     1. **Text Classification (e.g., sentiment analysis).** Here the goal is to obtain the sentiment of a review (e.g., positive or negative), detecting if an email is spam, or determining if a sentence is grammatically correct. Here,  the input is a text, and the output variable is the class label we are looking for. This classification task is discriminative and supervised. 
      
             Input: The hotel I spent my time was not clean.
             Output: Negative/0.       
  
     2. **Named Entity Recognition.** The goal here is to find which parts of the input text correspond to entities such as persons, locations, or organizations. In this task, the input is a piece of text, and \\(\mathbf{y}\\), the output variable is the label of different entities in the input text. This classification task is discriminative and supervised. 
 
             Input: My name is John and I work at Microsoft in Seattle.
             Output: John is a person (PER), Microsoft is an organization (ORG), and Seattle is a location (LOC).
  
     3. **Question Answering.** Here, we are looking for answering questions using information from a given context. The input is a text (context), including some information for the answer, and another text for the question. The output variable is another text with the answer to the asked question. This is a discriminative/generative task and typically a supervised task.
        
             Input: Question="Where do I work?, context: My name is John and I work at Microsoft in Seattle.
             Output: {Score': 0.78, 'Start': 30, 'End': 38, 'Answer': 'Microsoft'}
        
     5. **Translation.** Translating from one language (e.g., English) to another language (e.g., Persian). The input is a text from the source language and the output is the desired translation. 
    
             Input: How are you doing, Ali?
             Output: حالت چطورهست، علی ؟

     7. **Summarization.** Summarization is the task of reducing a text into a shorter text while keeping most of the important parts of referenced in the text. Both input and output are texts. This may be a supervised and discriminative task.

             Input: America has changed dramatically in recent years. Not only has the number
                   of graduates in traditional engineering disciplines such as mechanical, civil, 
                   electrical, chemical, and aeronautical engineering declined, but in most of 
                   the premier American universities engineering curricula now concentrate on 
                   and encourage largely the study of engineering science. As a result, there 
                   are declining offerings in engineering subjects dealing with infrastructure, 
                   the environment, and related issues, and greater concentration on high 
                   technology subjects, largely supporting increasingly complex scientific 
                   developments. While the latter is important, it should not be at the expense 
                   of more traditional engineering.
             Output: America has changed dramatically in recent years. The 
                     the number of engineering graduates in the U.S. has declined in 
                     traditional engineering disciplines such as mechanical, civil,
                     electrical, chemical, and aeronautical engineering.
        
     9. **Text Generation.** Here, by providing a prompt, a model can auto-complete a piece of text by generating the remaining of it. In this task, \\(\mathbf{x}\\), the input variable is a piece of text, and the output variable is some other text related to the input. This task is generative and supervised/unsupervised.

             Input: In this course, we will teach you how to
             Output: In this course, we will teach you how to understand and use data flow and data interchange when handling user data.
  
     10. **Fill-Mask.** Here, the goal is to fill in the blanks in a given text. The input is a text with a word(s) masked and the output is the desired missing word. This task is typically a generative task and supervised.
     
             Input: This course will teach you about <mask> models.
             Output: <mask> = Machine learning


   * **Tasks in Speech/Audio.** The input data for this domain include audio and speech files, coming from recorded speeches, meetings, movies, news, call centers, etc.
     
     1. **Automatic Speech Recognition (ASR).** The goal of ASR is to transcribe speech audio recordings into text. ASR task has many practical applications, from creating closed captions for videos to enabling voice commands for virtual assistants. The input data is an audio file and the output is its transcription text. This is a supervised and discriminative task.

        <p align="center">
        <img width="700" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/a4a6743e-959a-46f2-9c76-e796cd05de71">
             <br>
             <em>Speech-to-Text (ASR).</em>
        </p>
        
     2. **Text-to-Speech.** This is the reverse task of ASR. Given a text, we want to generate a similar human speech. This is typically a supervised and discriminative task.
          
          <p align="center">
          <img width="690" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/349a30b0-69bf-4edd-a025-f83f407963f1">
          <br>
             <em>Text-to-Speech.</em>
        </p>
  
     3. **Audio-to-Audio.** This is a family of tasks in which the input is audio, and the output is one or multiple generated audios. Some example tasks are **speech enhancement** and **Source Separation (SS).**. These tasks are regression tasks and can be discriminative/generative and supervised/unsupervised.
         - In Source Separation, the goal is to separate multiple speeches from their superposition.
        
          <p align="center">
          <img width="800" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/8ffd1f55-79a6-427f-b284-a3664a866bdf">
          <br>
             <em>Speech Separation (SS). The input is the superposition of two sources, and the output contains two sources individually.</em>
          </p>
 
          - In Speech Enhancement, the goal is to clean (denoise) audio, so it can be heard and possibly understood in a better way.
       
           <p align="center">
           <img width="800" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/7f6100ae-02a9-41e3-be19-28fc4eaca7ee">
          <br>
             <em>Speech Enhancement. The input is noisy audio, and the output is the clean (denoised) version of the input.</em>
          </p>

          
     4. **Voice Activity Detection (VAD).**- In VAD, the goal is to determine which part of a speech signal is an actual speech (it is not silent, noise, etc.). The input is the speech signal, and the output is a binary mask for each time sample (0-1 signal in time). This is typically a supervised and discriminative classification task.
       
         <p align="center">
          <img width="757" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/66a738f4-5d74-4834-b6b3-607958216b4e">
          <br>
             <em>Speech separation. The input is the superposition of two sources, and the output contains two sources individually.</em>
          </p>
  

     5. **Speech Diarization.** Who spoke when? Speaker diarization is the task of splitting audio based on the speaker's identity.
    
         <p align="center">
          <img width="1200" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/6efeeec7-d262-4a38-9da0-fc654ade0553">
          <br>
             <em>Speech Diarization [https://www.assemblyai.com/].</em>
          </p>

     6. **Intent Classification.** Here, the goal is to classify the input speech according to its intent. Typically, this task is modeled as a classification task which labels each input audio sample with a set of non-binary properties as shown in the following figure.

          <p align="center">
          <img width="757" alt="Screenshot 2023-07-17 at 9 49 36 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/89e59100-738f-43fb-9051-55c96d483a7b">
          <br>
             <em>A multiclass-multioutput classification (aka multitask classification).</em>
          </p>

   * **Tasks in Multimodal data.** The input data for this type can be any of the above modalities or other things (e.g., time series).
     
     1. **Text-to-Image.** Here, the goal is to teach a model to create an image from a description of a given text. This is a generative task.
        
          <p align="center">
          <img width="690" alt="Screenshot 2023-07-18 at 10 58 05 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/b753ba77-1016-443d-9cd9-96e03a1ca55c">
           <br>
             <em>Text-to-Image [https://imagen.research.google/].</em>
          </p>

     2. **Image-to-Text.** This is the reverse of the previous task. The input is an image, and the output is a text describing the content of the image. Image captioning is a popular application of this generative/discriminative task.
 
         <p align="center">
          <img width="700" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/b9cfc7b4-d2af-418a-94d3-3490b1479fbf">
           <br>
             <em>Image-to-Text [A. Das and d S. Saha, 2020].</em>
          </p>
        
     4. **Visual Question Answering (VQA).** This is a task in computer vision that involves answering questions about an image. The goal of VQA is to learn a model to understand the content of an image and answer questions about it in natural language.
         <p align="center">
          <img width="700" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/cc054c7d-fb3c-4519-b80d-cd1e80c80a78">
            <br>
             <em>Visual Question Answering (VQA) [https://huggingface.co/tasks/visual-question-answering].</em>
          </p>

     5. **Graph Machine Learning (GML).** A graph is a description of items linked by relations such as social networks, molecules, and knowledge graphs. Graph machine learning (GML) is the application of machine learning to graphs specifically for predictive and prescriptive tasks. GML has a variety of use cases across the supply chain, fraud detection, recommendations, customer 360, drug discovery, etc. One of GML’s primary purposes is to compress large sparse graph data structures to enable feasible prediction and inference. GML can be accomplished either by supervised methods or using unsupervised approaches. One example of GML is to predict interactions or collaborations between employees in an organization. This is called _link prediction_ problem. Here, the objective is to predict whether there would be a link between any two unconnected nodes. This is a classification/regression supervised and discrimination task.

        <p align="center">
          <img width="776" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/47f5fcf8-01cf-410f-86f6-90f930e145d5">
          <br>
             <em>Predicting interactions between people. [https://www.analyticsvidhya.com/blog/].</em>
        </p>
       
   * **Other Tasks.** Here we consider two more tasks in RL and recommendation systems.
     
     1. **The Cart-Pole problem using Reinforcement Learning.** The Cart-Pole problem is an inverted pendulum problem where a rod/stick is balanced upright on a cart. The goal is to learn an agent/learner to keep the stick from falling over by moving the cart right or left. At each timestep, if the agent can hold the stick upright, it receives a positive reward (e.g., +1); otherwise, it receives a negative reward (e.g., -1). Using RL, The objective is to maximize the total reward by preventing the stick from falling over.
 
         <p align="center">
          <img width="600" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/0fb960ef-95b6-49e2-9864-7c965fc2d511">
         <br>
             <em>A Cart-Pole Problem. We can move the cart either to the left or right. When we move the cart to the left, the pole tends to fall to the right side, and vice versa for the other direction.</em>
         </p>

     2. **Netflix Movie Recommendation Problem.** In 2006, Netflix challenged the machine learning communities to develop an algorithm that could show the better accuracy of its recommendation system, Cinematch. Netflix released a dataset of 100 million anonymous movie ratings for this challenge and offered one million dollars for improving the accuracy of Cinematch by 10%. The following figure shows a schematic of the Netflix dataset. The green checks denote the available reviews for the users and movies. The challenge here is to predict (some of) empty entries (tose with "?" sign) without any vote. This problem is also called _Matrix completion_ as the goal is to complete empty entries of a data matrix using other filled/given entries. This problem is a supervised and discriminative task. 
    
         <p align="center">
          <img width="600" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/1e3e8eed-0ae2-4616-a7fd-06cade2bebb0">
         <br>
             <em>Reviewing 5 movies by 4 users. Given available reviews what will the votes be for "?".</em>
         </p>

      3. **Predicting Stock Prices.** This is a common problem in many financial companies like Fidelity. Given the stock price of a company in some period, can we predict the stock price in the future time? The following figure shows such a task using Tesla stock price.

         <p align="center">
          <img width="700" alt="Screenshot 2023-07-18 at 11 21 41 PM" src="https://github.com/mrezasoltani/mrezasoltani.github.io/assets/37226975/b69ba691-1a71-4bad-8fe5-27ee164cddc6">
         <br>
             <em>Predicting Tesla stock price [https://towardsdatascience.com/lstm-time-series-forecasting-predicting-stock-prices-using-an-lstm-model-6223e9644a2f].</em>
         </p>
