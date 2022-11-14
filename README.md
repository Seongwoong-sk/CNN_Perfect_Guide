# 딥러닝 CNN 완벽 가이드 - Fundamental편

**Inflearn**에서 제공하는 [딥러닝 CNN 완벽 가이드 - Fundamental편](https://www.inflearn.com/course/%EB%94%A5%EB%9F%AC%EB%8B%9D-cnn-%EC%99%84%EB%B2%BD-%EA%B8%B0%EC%B4%88/dashboard)을 공부하고 정리한 문서입니다.    

**Framework**는 **Tensorflow 2.x**로 진행되었습니다.

      
## 목차


### Section I. Introduction to Deep Learning and Gradient Descent

- [x] 머신러닝의 이해
- [x] 딥러닝 개요, 장단점과 특징
- [x] 퍼셉트론의 개요와 학습 메커니즘 이해
- [x] 회귀 개요와 RSS, MSE의 이해
- [x] 경사하강법, 경사하강법을 이용한 선형회귀 구현, 
- [x] 확률적 경사하강법(Stochastic Gradient Descent)구현
- [x] 미니 배치(Mini-batch) 경사하강법 구현


### Section II. BackPropagation, Activation Function, Loss Function, Optimizer

- [x] 오차 역전파(BackPropagation)의 이해, 미분의 연쇄 법칙
- [x] BackPropagation의 Gradient 적용 메커니즘
- [x] 활성화 함수(Activation Function)의 이해
- [x] 손실 함수(Loss Function)과 Cross Entropy
- [x] Optimizer의 이해 - Momentum, AdaGrad, RMSProp, Adam


### Section III. Keras Framework

- [x] 이미지 배열의 이해
- [x] Dense Layer - Fashion MNIST
- [x] Keras Layer API
- [x] Functional API
- [x] Keras CAllback


### Section IV. CNN의 이해

- [x] Feature Extractor
- [x] Computation of Convolutional Neural Network
- [x] Kernel , Feature Map
- [x] Stride , Padding , Pooling
- [x] Convolutional Neural Network of Multi-Channel Input 
- [x] Calculation of size of feature map when applying CNN


### Section V. CNN 모델 구현 및 성능 향상 기본 기법 적용하기 

- [x] CIFAR10 데이터세트를 이용하여 CNN 모델 구현 실습
- [x] 가중치 초기화(Weight Initialization)의 이해와 적용
- [x] 배치 정규화(Batch Normalization)의 이해와 적용
- [x] 학습 데이터 Shuffle 적용 유무에 따른 모델 성능 비교
- [x] 배치 크기 변경, 학습율(Learning Rate) 동적 변경에 따른 모델 성능 비교
- [x] 필터수와 층 (Layer) 깊이 변경에 따른 모델 성능 비교
- [x] Global Average Pooling의 이해와 적용
- [x] 가중치 규제(Weight Regularization)의 이해와 적용



### Section VI. 데이터 증강의 이해 - Keras ImageDataGenerator 활용

- [x] 데이터 증강(Data Augmentation)의 이해
- [x] Keras의 ImageDataGenerator로 Augmentation 적용 -> CIFAR10



### Section VII. Pretrained CNN 모델의 활용과 Keras Generator 메커니즘 이해

- [x] Pretrained Model의 이해와 Transfer Learning 개요
- [x] Pretrained VGG16, Xception
- [x] Cat and Dog Image Classification 
- [x] Major Python Image Libraries
- [x] Keras Generator 기반의 Preprocessing과 Data Loading Mechanism
- [x] flow_from_directory()


### Section VIII. Albumentation을 이용한 Augmentation기법과 Keras Sequence 활용하기

- [x] Albumentations 사용 - Flip, Shift, Scale, Rotation, Crop, Bright, Contrast, HSV, Noise, Cutout, CLAHE, Blur, Oneof etc 
- [x] Keras Sequence Class, Dataset Generation based on Sequence
- [x] Image Classification utilizing Xception, MobileNet, Albumentations, Dataset based on Sequence


### Section IX. Advanced CNN 모델 파헤치기 - AlexNet, VGGNet, GooLeNet

- [x] 역대 주요 CNN 모델들의 경향과 특징
- [x] AlexNet의 개요와 구현 코드 이해 - CIFAR10으로 학습 및 성능 테스트
- [x] VGGNet의 이해, 구조 상세 및 구현코드 이해하기
- [x] VGGNet16 모델 직접 구현하기
- [x] GoogLeNet(Inception) 구조 상세 및 구현 코드 이해
- [x] 1x1 Convolution

### Section X. Advanced CNN 모델 파헤치기 - ResNet 상세와 EfficientNet 개요

- [x] ResNet의 이해 - 깊은 신경망의 문제와 identity mapping, Residual Block
- [x] ResNet 아키텍처 구조 상세
- [x] ResNet 모델 직접 구현하기 - 구현한 ResNet50 모델로 CIFAR10 학습 및 성능 테스트
- [x] EfficientNet의 이해와 아키텍처


### Section XI. Fine Tuning of Pretarined Model & 다양한 Learning Rate Scheduler의 적용

- [x] 사전 훈련 모델의 미세 조정(Fine Tuning) 학습 이해와 수행
- [x] Learning Rate Scheduler & Keras LearningRateScheduler Callback
- [x] Cosine Decay와 Cosine Decay Restart 이해와 적용
- [x] Ramp Up and Step Down Decay 이해와 적용


### Section XII. 종합 실습 1 -120종의 Dog Breed Identification 모델 최적화

- [x] Dog Breed Identitcation 데이터 세트 특징과 모델 최적화 개요
- [x] Dog Breed 데이터의 메타 DataFrame 생성 및 이미지 분석, Sequence 기반 Dataset 생성
- [x] Xception 모델 학습, 성능평가 및 예측 후 결과 분석
- [x] EfficientNetB0 모델 학습, 성능평가 및 분석
- [x] Augmentation과 Learning Rate 최적화
- [x] Pretrained 모델의 Fine-Tuning을 통한 모델 최적화
- [x] Config Class 기반으로 함수 변경 후 EfficientNetB1 모델 학습 및 성능 평가



### Section XIII. 종합 실습 2 - 캐글 Plant Pathology(나무잎 병 진단) 경연 대

- [x] Plant Pathology 캐글 경연대회 및 데이터 세트 가공
- [x] Augmentation 적용 분석, Sequence기반 Dataset 생성
- [x] Xception 모델 학습 후 Kaggle에 성능 평가 csv 파일 제출하기
- [x] 이미지 크기 resize 후 Xception 모델 학습 및 성능 평가
- [x] EfficientNetB3,B5 and B7 모델 학습 및 성능 평가
