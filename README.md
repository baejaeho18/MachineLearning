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

**1) Task1** : 모델링할 함수를 알아내기 위해 이를 근사할 수 있는 근사 식이 필요하다.
$$y(x, w) = w_0 + w_1x + w_2x^2 + \ldots + w_Mx^M = \sum\limits_{i=0}^M w_ix^i$$
* 테일러 급수 : 차수(M)을 늘릴수록 특정 위치에서의 함수를 잘 근사 - 일반화에는 약간 거리가 있음
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

**5) Task5** : Overfitting 문제를 해결한다.
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
* product rule : $p(X,Y) = p(Y|X)p(X)$
* Bayesian rule : $p(Y|X) = \frac{p(X|Y)p(Y)}{p(X)}$
### 0.2.1 Desity Estimation

### 0.2.2 Expactations and Covariances

### 0.2.3 Bayesian Probabilities

### 0.2.4 Gaussian Distribution


## 0.3 Information Theory - 와 진짜 재밌다 하.하.하!
정보(information)? "학습에 있어 필요한 놀람의 정도(degree of surprise)"
$$h(x) = -log_2p(x)$$
엔트로프(entropy)? 평균 정보량이자 p(x)인 분포에서 h(x) 함수의 기댓값
$$H[x] = -\sum_x p(x)log_2p(x)$$
- 밑수가 2라면 정보량의 단위는 bit라고 보면 된다.
- Non-Uniform 분포의 엔트로피는 Uniform 분포의 엔트로피보다 낮다.
ex) a,b,c,d,e,f,g,h 8글자의<br>
확률분포가 동일($\frac{1}{8}$)할 경우 : $H[x] = -8 * \frac{1}{8}log_2\frac{1}{8} = 3 bits$ <br>
확률분포가 ( $\frac{1}{2} \, \frac{1}{4} \, \frac{1}{8} \, \frac{1}{16} \, \frac{1}{32} \, \frac{1}{64} \, \frac{1}{64}$ )일 경우 : <br>
$H[x] = -\frac{1}{2}log_2\frac{1}{2} - \frac{1}{4}log_2\frac{1}{4} - \frac{1}{8}log_2\frac{1}{8} - \frac{1}{16}log_2\frac{1}{16} - \frac{1}{32}log_2\frac{1}{32} - \frac{1}{64}log_2\frac{1}{64} - \frac{2}{128}log_2\frac{1}{128}= 2bits$ <br>
$E[length] = \frac{1}{2} * 1 + \frac{1}{4} * 2 + \frac{1}{8} * 3 + \frac{1}{16} * 4 + \frac{1}{32} * 6 + \frac{1}{64} * 6 + \frac{1}{128} * 6 + \frac{1}{128} * 6 = 2 bits$ <br>
(hoffman code : 0, 10, 110, 1110, 111100, 111101, 111110, 111111) <br>
즉, 엔트로피는 랜덤 변수의 상태를 전송하는데 필요한 비트 수의 Lower Bound이다.

## Entropy as Statistical Mechanics
통계역학 관점에서 엔트로피는 "어떤 계의 무질서도" 도는 "거시 상태에 대응되는 미시 상태의 가짓 수"로 표현된다. <br>
최대 엔트로피 값을 구하기 위해 라그랑지안 승수(제약 조건 상에서 함수의 최소값을 찾는 수단)을 사용한다.
### 미분 엔트로피(differential entropy) <br>
  - 입력 변수가 연속일 때, 가우시안 분포가 entropy를 최대로 만든다
  - 입력 변수가 이산일 때, 균일(uniform) 분포가 엔트로피를 최대로 만든다.
### 조건부 엔트로피(conditional entropy)
mutual information : y를 알고 난 후에 x의 불확실성을 줄이는 과정
### 연관 엔트로피(Relative entropy)
- Relative Entropy 또는 KL(Kullback-Leibler) divergence
  * Jensen's inequality

  즉, KL divergence를 최소화하는 것은 likelihood(가능도 함수)를 최대화 시키는 것과 동일하다 -> MLE와 연관성을 지님

## 0.4 Decision Theory
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
