# The Application of CNN and SIFT-SVM on Chest X-ray Images for COVID-19 Detection  

## 📖 Introduction  
This project applies **Convolutional Neural Networks (CNN)** and **SIFT + Support Vector Machine (SVM)** for the classification of chest X-ray images, aiming to detect COVID-19 infections.  

## ⚙️ Methodology  
- **CNN**:  
  - Feature extraction via convolution and pooling layers.  
  - Fully connected layers for binary classification.  
  - Tuned hyperparameters such as batch size, filter size, and regularization (L1/L2).  

- **SIFT + SVM**:  
  - Extracted scale-invariant features from X-ray images.  
  - Constructed bag-of-features (BoF) embeddings using K-means clustering.  
  - Applied SVM with radial basis function (RBF) kernel.  
  - Tuned cluster size (*K*) and regularization parameter (*C*).  

## 📊 Dataset  
- Source: [COVIDx CXR-2 dataset (Kaggle)](https://www.kaggle.com/andyczhao/covidx-cxr2?select=test.txt)  
- Training set: 2,158 positive + 2,158 negative images (balanced).  
- Test set: 400 images (200 positive, 200 negative).  
- Preprocessing: resized to **300×300**, random rescaling, flipping, zooming.  

## 🚀 Implementation  
- **CNN**:  
  - ReLU activation, 3×3 filters, max pooling, dense layers.  
  - Compared batch sizes (32 vs 64) and structural variations.  

- **SIFT + SVM**:  
  - OpenCV for SIFT feature extraction.  
  - Scikit-learn for K-means clustering and SVM classification.  

## 📈 Results  
- **CNN**:  
  - Best model achieved **90.75% test accuracy** with batch size = 32.  
  - Smaller batch size improved generalization, though more time-consuming.  

- **SIFT + SVM**:  
  - Best performance at **C = 1, K = 250**.  
  - Faster to train but slightly lower accuracy (~85–88%).  

- **Comparison**:  
  - CNN outperforms SIFT-SVM in accuracy.  
  - SIFT-SVM is more efficient in training time.  

## ✅ Conclusion  
- CNN is more accurate (~90%) but computationally intensive.  
- SIFT-SVM offers a lighter, faster alternative with competitive performance.  
- Both methods demonstrate the feasibility of chest X-ray–based COVID-19 detection.  

## 📚 References  
1. [Simple Guide to Hyperparameter Tuning in Neural Networks](https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594)  
2. [COVIDx CXR-2 Dataset on Kaggle](https://www.kaggle.com/andyczhao/covidx-cxr2?select=test.txt)  
3. [Medical X-ray Image Classification with CNN](https://towardsdatascience.com/medical-x-ray-%EF%B8%8F-image-classification-using-convolutional-neural-network-9a6d33b1c2a)  
4. [SIFT + SVM for Classification](https://liverungrow.medium.com/sift-bag-of-features-svm-for-classification-b5f775d8e55f)  
5. [Introduction to SIFT (OpenCV Docs)](https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)  
