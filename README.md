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
* Decision Theory : 확률적 표현을 바탕으로 적절한 기준에 따라 최적의 예측을 수행할 수 있는 방법론을 제공 

**1) Task1** : 모델링할 함수를 알아내기 위해 이를 근사할 수 있는 근사(Approximate)식이 필요하다.
$$y(x, w) = w_0 + w_1x + w_2x^2 + \ldots + w_Mx^M = \sum\limits_{i=0}^M w_ix^i$$
* Tayler seires : 차수(M)을 늘릴수록 특정 위치에서의 함수를 잘 근사. 일반화에는 약간 거리가 있음 [[blog]](https://darkpgmr.tistory.com/59)
  $$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \ldots$$
* 퓨리에 변환식
  $$F(w) = \int_{-\infty}^{\infty} f(t)e^{-iwt}dt$$

**2) Task2** : 식을 근사하기 위해 $E(w) = y(x,w) - t$를 최소화하는 최적의 계수 W를 구한다.
* 에러 함수(Error Function) : 제곱합(sum-of-squares) 에러 함수를 주로 사용
  $$E(w) = \frac{1}{2}\sum\limits_{n=1}^N ({y(x_n,w)-t_n})^2$$

**3) Task3(model selection)** : 최적의 차수(M)의 개수를 찾는다.
M이 너무 작으면 under-fitting, 너무 높으면 over-fitting 현상이 발생한다.

**4) Task4** : 모델의 평가를 위해 에러 값의 정도를 수치화한다.
$$E_{RMS} = \sqrt{\frac{2E(w^*)}{N}} (N:sample\ size)$$
* N으로 나누는 것은 데이터셋 크기가 다른 스케일 문제를 보정하기 위한 정규화(Normalization) 요소임

**5) Task5** : Over-fitting 문제를 해결한다.
* 관찰 데이터의 크기가 클수록 over-fitting 현상이 줄어든다.
그러나 휴리스틱 관점에서, 모델 파라미터 개수(M)는 샘플 크기의 1/5, 1/10 정도보다 작은 것이 좋다.
* Regularization : w값의 범위를 제한하여 모델의 복잡도(M,차수)는 올리면서 over-fitting을 막아내는 것
  $$E(w) = \frac{1}{2}\sum\limits_{n=1}^N ({y(x_n,w)-t_n})^2+\frac{\lambda}{2}\|\|w\|\|^2 \ (\|\|w\|\|^2 = w^Tw)$$
* MLE, Bayesian 등으로 적절한 M 값을 찾아내는 방식도 존재함
  
## 0.2 Probability Theory
불확실성을 정량화시켜 표한할 수 있는 수학적인 프레임워크를 제공한다.
불확실성(Uncertainty)은 데이터 크기의 부족 혹은 노이즈(Noise)로 인해 발생한다. <br>
확률의 법칙
* sum rule : $p(X) = \sum\sigma_{Y} p(X,Y)$
* product rule : $p(X,Y) = p(Y|X)p(X)$ (if 
* Bayesian rule : $p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$ [[blog]](https://roytravel.tistory.com/350)
### 0.2.1 Desity Estimation
밀도 추정(Density Estimation) : 주어진 데이터x로부터 확률 밀도 함수(probability density function, PDF)를 추정하는 과정
* 확률이 이산(discrete)적인 사건
  - Histogram
  - PMF(Probability Mass Function) : 각 데이터 포인트에 대한 상대 빈도를 계산
* 확률이 연속일 경우
  - KDE(Kernel Density Estimation) : 확률값을 구간(range)로 표현한다. 구간 적분해야 확률값이 된다.
### 0.2.2 Expactations and Covariances
* 평균(Expactation)
  - $E[f] = \sum_x p(x)f(x)$ (이산)
  - $E[f] = \int p(x)f(x) dx$ (연속)
* 조건부 평균(기댓값) : $E_x[f|y] = \sum_x p(x|y)f(x)$
* 분산(variance) : $var[f] = E[f(x) - E[f(x)])^2] = E[f(x)^2] - E[f(x)]^2$
* 임의의 변수 x,y에 대한 공분산(covariance)
  - cov[x,y] = E_{x,y}[(x - E[x])(y - E[y])] = E_{x,y}[xy] - E[x]E[y] (실수)
  - cov[x,y] = E_{x,y}[(x - E[x])(y^T - E[y^T])] = E_{x,y}[xy^T] - E[x]E[y^T] (벡터)

### 0.2.3 Bayesian Probabilities
실제 측정된 빈도수에 따라 확률을 계산하는 Frequentism과 다르게 Baysianism은 믿음의 정도로 불확실성을 정량화할 수 있다. MLE(Maximun Likelihood Estimation)과 달리, 파라미터 w를 랜덤변수로 간주하여 확률분포를 사용한다. 
$$p(w|D) = \frac{p(D|w)p(w)}{p(D)}$$
관찰되는 데이터가 존재하기 이전에 이미 p(w)를 통해 w의 불확실한 정도(prior)를 수식에 반영하고 있다.
이후 실제 데이터를 통해 예측된 w의 확률(likelihood)를 조합하여 실제 w의 사후 확률(posterior)을 기술한다.
사전확률로 인해 덜 극단적인 결과를 얻도록 보정된다.
예를 들어, 사후 확률이 최대가 되는 파라미터 값을 찾는 MAP(Maximum A Posteriori) 추정은 데이터가 적거나 불확실한 경우 사전 분포를 이용해 추정값을 더욱 안정적으로 만들 수 있다.

### 0.2.4 Gaussian Distribution
$$N(x|\mu, \sigma^2) = \frac{1}{(2\pi\sigma^2)^{1/2}}exp({-\frac{1}{2\sigma^2}(x-\mu)^2})$$
- 2개의 파리미터(모수,parameter) : $\mu$:평균(mean), $\sigma$:표준편차(standard deviation)
- 정확도(precision) : 분산(표준편차 제곱)의 역수  $\beta = \frac{1}{\sigma^2}$
- 가우시안 분포, 즉 정규분포의 장점은 1차원의 속성을 D차원으로 확장 가능하다는 것이다. 이를 다변량(multinominal) 가우시안 분포라고 부른다.
  $$N(x|\mu, \Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)) (\Sigma:covariance:DxD)$$
- 주어진 데이터가 어떤 가우시안 분포를 따르는지 해를 구하려면 **평균과 분산**(파라미터)를 추정하면 된다.
  $$\mu_{MLE} = \frac{1}{N} \sum\limits_{n=1}^N x_n \quad \sigma_{MLE}^2 = \frac{1}{N} \sum\limits_{n=1}^N {(x_n - \mu_{MLE})}^2$$
- MLE에서 발생하는 over-fitting의 한 예는 분산값의 bias이다.
모집단의 샘플 그룹들에 대해 각각의 평균 값을 평균하면 실제 평균값과 가까워진다. 그러나 각각의 분산 값을 평균해도 실제 분산 값에 가까워지진 않는다.
  ![image](https://github.com/baejaeho18/MachineLearning/assets/37645490/c0d493fe-d104-4273-8536-cae55cf32b8a)

### 0.2.5 Non-parametic Methods
multi-modal D차원 상의 데이터는 N번 관찰 시 V크기의 부피공간 영역R에 속하는 데이터의 수K로 확률밀도함수p(x)를 표현할 수 있다.
$$P = \int_R p(x) dx = p(x)V \quad p(x) = \frac{K}{NV}$$
* KDE(Kernel Density Estimators) : V를 고정시키고 K를 결정
  - 커널 함수(kernel method) : 임의의 한 점 x가 주어졌을 때, 각 차원으로 h 거리 내에 존재하는 모든 샘플을 센다.
  - smoothing : 연속적인 확률분포를 구성하기 위해 하나의 샘플$x_n$에 대해 중심이 $x_n$이고 표준편차가 h인 정규분포를 만들고 이를 합하여 새로운 밀도함수를 만들어내는 것.
  - h값이 충분히 크지 않으면 under-smooth, 충분히 작지 않으면 over-smooth가 발생한다.
* KNN(K-NEarest Neighbors) : K를 고정시키고 V를 결정
  - 샘플 데이터 $x_n$을 중심으로 하는 구(sphere)가 K개의 샘플을 포함할 때까지 구의 반지름을 늘려, 구의 부피V와 확률밀도p(x)를 구한다.
  - 주로 분류(classification) 문제에 사용된다.
  - K값으로 smoothing 정도를 조절할 수 있다.

## 0.3 Information Theory - 와 진짜 재밌다 하.하.하!
정보(information) : "학습에 있어 필요한 놀람의 정도(degree of surprise)"
  $$h(x) = -log_2p(x)$$
엔트로피(entropy) : "평균 정보량"이자 p(x)인 분포에서 h(x) 함수의 기댓값
  $$H[x] = -\sum_x p(x)log_2p(x)$$
- 밑수가 2라면 정보량의 단위는 bit라고 보면 된다.
- Non-Uniform 분포의 엔트로피는 Uniform 분포의 엔트로피보다 낮다. <br>
ex) a,b,c,d,e,f,g,h 8글자의<br>
확률분포가 동일($\frac{1}{8}$)할 경우 : $H[x] = -8 * \frac{1}{8}log_2\frac{1}{8} = 3 bits$ <br>
확률분포가 ($\frac{1}{2} \, \frac{1}{4} \, \frac{1}{8} \, \frac{1}{16} \, \frac{1}{32} \, \frac{1}{64} \, \frac{1}{64}$)일 경우 : <br>
$H[x] = -\frac{1}{2}log_2\frac{1}{2} - \frac{1}{4}log_2\frac{1}{4} - \frac{1}{8}log_2\frac{1}{8} - \frac{1}{16}log_2\frac{1}{16} - \frac{1}{32}log_2\frac{1}{32} - \frac{1}{64}log_2\frac{1}{64} - \frac{2}{128}log_2\frac{1}{128}= 2bits$ <br>
$E[length] = \frac{1}{2} * 1 + \frac{1}{4} * 2 + \frac{1}{8} * 3 + \frac{1}{16} * 4 + \frac{1}{32} * 6 + \frac{1}{64} * 6 + \frac{1}{128} * 6 + \frac{1}{128} * 6 = 2 bits$ <br>
(hoffman code : 0, 10, 110, 1110, 111100, 111101, 111110, 111111) <br>
즉, 엔트로피는 랜덤 변수의 상태를 전송하는데 필요한 비트 수의 Lower Bound이다.
### Entropy as Statistical Mechanics
통계역학 관점에서 엔트로피는 "어떤 계의 무질서도" 도는 "거시 상태에 대응되는 미시 상태의 가짓 수"로 표현된다. <br>
최대 엔트로피 값을 구하기 위해 라그랑지안 승수(제약 조건 상에서 함수의 최소값을 찾는 수단)을 사용한다. [[blog]](https://velog.io/@nochesita/%EC%B5%9C%EC%A0%81%ED%99%94%EC%9D%B4%EB%A1%A0-%EB%9D%BC%EA%B7%B8%EB%9E%91%EC%A3%BC-%EC%8A%B9%EC%88%98%EB%B2%95-Lagrange-Multiplier-Method)
* 미분 엔트로피(differential entropy) <br>
  - 입력 변수가 연속일 때, 가우시안 분포가 entropy를 최대로 만든다
  - 입력 변수가 이산일 때, 균일(uniform) 분포가 엔트로피를 최대로 만든다.
* 조건부 엔트로피(conditional entropy) <br>
mutual information : y를 알고 난 후에 x의 불확실성을 줄이는 과정
* 연관 엔트로피(Relative entropy)
  - Relative Entropy 또는 KL(Kullback-Leibler) divergence
  - Jensen's Inequality for convex function [[blog]](https://blog.naver.com/PostView.naver?blogId=sw4r&logNo=221166257113)
$$\ if\ \phi(E[X]) \leq E[\phi(X)] \, \phi(x)\ is\ convex$$
  즉, KL divergence를 최소화하는 것은 likelihood(가능도 함수)를 최대화 시키는 것과 동일하다 -> MLE와 연관성을 지님 <br>

## 0.4 Decision Theory
확률적 표현을 바탕으로 적절한 기준에 따라 최적의 예측을 수행할 수 있는 방법론을 제공한다.
Loss function(know as Cost function) : 하나의 샘플x가 실제로는 특정 클래스 $C_k$에 속하지만, 모델이 이 샘플의 클래스를 $C_j$로 선택할 때 들어가는 비용을 정의
$$E[L] = \sum_k\sum_j\int_{R_j}L_{kj}[(x,C_k)dx$$

Question? 주어진 label이 거짓일 경우를 모델이 고려할 수 있는가? 


# 1. Clustering : Unsupervised Learning
## Dimension Reduction
## Nonlinear Dimension Reduction

# 2. Classification : Supervised Learning
## 2.1 Discriminant Functions
## 2.2 Laplace Approximation
## 2.3 Bayesian Logistic Regression

# 3. Ensemble Learning

# 3. Regression : Supervised Learning
## 3.1 Linear Regression
## 3.2 MLE and LSE
### Maximum Likelihood
### Least Squares
## 3.3 Bias-Variance Decomposition
## 3.4 Bayesian Linear Regression

# 4. Neural Netowrks
## 4.1 Feed-forward Network
## 4.2 Network Training
## 4.3 Error Backpropagation
## 4.4

# 5. Numerical Optimization

# 6. Regularization

# 7. Deep Learning

## Recommender System

## SVM

## EM?

[[blog]](https://norman3.github.io/prml/)
## HMM?
