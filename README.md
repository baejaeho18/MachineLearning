# 0. Machine Learning

## 0.1 AI History
### Approaches
<img width="400" alt="image" src="https://github.com/baejaeho18/MyLibrary/assets/37645490/0468952e-3c6b-4ef8-822b-346f04bf8860"> <br>
### Timeline
Alan Turing : Turing machine and Turing test
<br>Neural Networks(Donald Hebb, 1949)
<br>Perceptron(Frank Rosenblatt, 1958) <-> (Marvin Minsky, 1969)
<br>Multiplayer Perceptron(1986)
<br>SVM(1995)
<br>Deep Neural Networks(Geoffrey Hinston, 2006)

## Introduction : What is ML?
### Components of ML
<img width="600" alt="image" src="https://github.com/baejaeho18/MyLibrary/assets/37645490/61c2e113-4ef7-4253-a9a5-6aed39e39122"> <br>
### Categories of ML
* Unsupervised Learning : clustering, dimension reduction by Density estimation, Pretraining
* Supervised Learning : speech/face recognition by Classification, Regression
  - Pattern Analysis(CSE@TAMU Gutierrez-Osuna) [[slide]](https://people.engr.tamu.edu/rgutier/lectures/pr/pr_l1.pdf)
* Semi-supervised Learning
* Reinforcement Learning : alphago, self-driving, machine translation
  - credit assignment problem
  - trade-off between exploration and exploitation
### Interdisciplinary
<img width="300" alt="image" src="https://miro.medium.com/v2/resize:fit:1210/format:webp/1*IFwhXu-_-LnlUJTbPxCq8Q.png"> <br>
### Bayseian Perceptive
$$p(w|D) = \frac{p(D|w)p(w)}{p(D)} (w: parameter, D: data)$$
If you don't know, goto Calculas for Linear algebra, probability and statistics.
<br>Data is given as a vector, matrix or tensor.
<br>Everthing is probabilistic due to noise.
<br>A single point does not say much, so statistics comes to rescue.
### What is DL?
Seminar for Deep Learning:Learning Based on deep neural network [[slides]](https://www.slideshare.net/TundeAjoseIsmail/deep-learning-presentation-102934185) <br>

## 0.1 Calculus and Linear Algebra
Object : 학습 데이터를 이용하여 모델을 근사한 후, 임의의 입력 데이터 x에 대해 예측결과 t를 제공
* Probablility Theory : 불확실성을 정량화시켜 표현할 수 있는 수학적인 프레임워크를 제공
* Decision Theory : 확률적 표현을 바탕으로 적절한 기준에 따라 최적의 예측을 수행할 수 있는 방법론을 제공 <br>
Task1 : 모델링할 함수를 알아내기 위해 이를 근사할 수 있는 근사 식이 필요하다.
$$y(x, w) = w_0 + w_1x + w_2x^2 + \ldots + w_Mx^M = \sum\limits_{i=0}^M w_ix^i$$
* 테일러 급수 : 차수(M)을 늘릴수록 특정 위치에서의 함수를 잘 근사 - 일반화에는 약간 거리가 있음
  $$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \ldots$$
* 퓨리에 변환식
  $$F(w) = \int_{-\infty}^{\infty} f(t)e^{-iwt}dt$$
Task2 : 식을 근사하기 위해 $E(w) = y(x,w) - t$를 최소화하는 최적의 계수 W를 구한다.
* 에러 함수(Error Function) : 제곱합(sum-of-squares) 에러 함수를 주로 사용
  $$E(w) = \frac{1}{2}\sum\limits_{n=1}^N ({y(x_n,w)-t_n})^2$$
Task3(model selection) : 최적의 차수(M)의 개수를 찾는다.
M이 너무 작으면 under-fitting, 너무 높으면 over-fitting 현상이 발생한다.
Task4 : 모델의 평가를 위해 에러 값의 정도를 수치화한다.
$$E_{RMS} = \sqrt{\frac{2E(w^*)}{N}} (N:sample\ size)$$
* N으로 나누는 것은 데이터셋 크기가 다른 스케일 문제를 보정하기 위한 정규화(Normalization) 요소임
Task5 : Overfitting 문제를 해결한다.
* 관찰 데이터의 크기가 클수록 over-fitting 현상이 줄어든다.
그러나 휴리스틱 관점에서, 모델 파라미터 개수(M)는 샘플 크기의 1/5, 1/10정도 보다 작은 것이 좋다.
* 모델의 복잡도는 올리면서(높은 차수) over-fitting을 막아내는 방법은, w가 취할 수 있는 값의 범위를 제한하는 것(Regularization)이다.
  $$E(w) = \frac{1}{2}\sum\limits_{n=1}^N ({y(x_n,w)-t_n})^2+\frac{\lambda}{2}\|\|w\|\|^2 \ (\|\|w\|\|^2 = w^Tw)$$

## 0.2 Probability
불확실성을 정량화시켜 표한할 수 있는 수학적인 프레임워크를 제공한다.
## 0.3 Desity Estimation

## 0.4 Information Theory

## 0.5 Decision Theory
확률적 표현을 바탕으로 적절한 기준에 따라 최적의 예측을 수행할 수 있는 방법론을 제공한다.


# 1. Clustering : Unsupervised Learning

## Dimension Reduction

## Nonlinear Dimension

# 2. Classification : Supervised Learning

# 3. Regression : Supervised Learning

# 3. Ensemble Learning

# 4. Neural Netowrks

# 5. Optimization

# 6. Regulation

# 7. Deep Larning

## SVM

## Recommend

## HMM
