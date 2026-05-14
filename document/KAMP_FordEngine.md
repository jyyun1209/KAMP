---
tags: [KAMP, ML, Classification, SensorData, FaultDetection]
---

> [!link] 관련 문서
> - [[데이터 상관 관계 활용 방법]] — Feature 선별·중복 제거·PCA 기법
> - [[분류 모델의 평가]] — ROC Curve, AUC 개념 정리
> - [[Terminology#러닝 커브]] — Learning Curve 용어 정의 (이수페타시스 용어집)

가이드북: [[Guidebook_FordEngine.pdf]]
구현 코드: https://github.com/jyyun1209/KAMP

# 개요

AI 분석 모델: 지도학습(Supervised Learning) 기반 알고리즘
- **분류(Classification)**: 특정 클래스에 속할 확률 계산 (불연속적 클래스 추정)
	- Linear Model: 로지스틱 회귀(Logistic Regression)
	- Ensemble: XGBoost
	- Neural Network: RNN, CNN
- 회귀(Regression): 종속 변수의 미래 값 예측 (연속적인 변수 추정)

![](Figures_MD/KAMP_FordEngine_FlowChart.png)

---
# 데이터 로딩 후, 클래스 균형 확인
![](Figures_MD/class_distribution.png)

---
# 상관 관계 분석
> 참고: [[데이터 상관 관계 활용 방법]]

| Pearson                             | Kendall                             | Spearman                             |
| ----------------------------------- | ----------------------------------- | ------------------------------------ |
| ![](Figures_MD/sensor_correlation_pearson.png) | ![](Figures_MD/sensor_correlation_kendall.png) | ![](Figures_MD/sensor_correlation_spearman.png) |

---
# 데이터 정규화

StandardScaler vs. **RobustScaler**(Outlier에 강건함)

**StandardScaler**
<img src="Figures_MD/normalization_standard.png" width="697">
**RobustScaler**
![](Figures_MD/normalization_robust.png)

---
# 모델 테스트
> 평가 지표 참고: [[분류 모델의 평가]]

## 1. Logistic Regression

| Confusion Matrix             | ROC Curve             | Learning Curve             |
| ---------------------------- | --------------------- | -------------------------- |
| ![](Figures_MD/confusion_matrix_lr.png) | ![](Figures_MD/roc_curve_lr.png) | ![](Figures_MD/learning_curve_lr.png) |
- 거의 찍는 것과 유사한 수준의 성능 (AUC: 0.476) → 선형 분류기(Linear Classifier)의 한계
- 샘플의 수가 증가하면 성능이 빠르게 감소 (Logistic Regression과 같이 단순한 모델일 때 나타날 수 있는 현상)
<img src="Figures_MD/training_loss_lr.png" width="697">
## 2. XGBoost

| Confusion Matrix              | ROC Curve              | Learning Curve              |
| ----------------------------- | ---------------------- | --------------------------- |
| ![](Figures_MD/confusion_matrix_xgb.png) | ![](Figures_MD/roc_curve_xgb.png) | ![](Figures_MD/learning_curve_xgb.png) |
![](Figures_MD/training_loss_xgb.png)
- 준수한 성능 (AUC: 0.852)
- 샘플 수 증가하면 성능 증가 경향
## 3. RNN (LSTM)

| Confusion Matrix              | ROC Curve              | Learning Curve              |
| ----------------------------- | ---------------------- | --------------------------- |
| ![](Figures_MD/confusion_matrix_rnn.png) | ![](Figures_MD/roc_curve_rnn.png) | ![](Figures_MD/learning_curve_rnn.png) |
![](Figures_MD/training_loss_rnn.png)
- 평범한 성능 (AUC: 0.785)
- 샘플 수 증가하면 성능 증가 경향
## 4. CNN

| Confusion Matrix              | ROC Curve              | Learning Curve              |
| ----------------------------- | ---------------------- | --------------------------- |
| ![](Figures_MD/confusion_matrix_cnn.png) | ![](Figures_MD/roc_curve_cnn.png) | ![](Figures_MD/learning_curve_cnn.png) |
![](Figures_MD/training_loss_cnn.png)
 - 매우 우수한 성능 (AUC: 0.970)
 - 샘플 수 증가하면 성능 증가 경향
---
# 모델 성능 개선 테스트

## 1. Logistic Regression

### A. 상위 Feature 선별
: 각 센서(500종)와 Label 간의 상관 관계를 분석하여, 강한 상관 관계를 가지는 센서들의 데이터만 선별하여 학습에 사용

| Pearson                                   | Kendall                                   | Spearman                                   |
| ----------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| ![](Figures_MD/sensor_label_correlation_pearson.png) | ![](Figures_MD/sensor_label_correlation_kendall.png) | ![](Figures_MD/sensor_label_correlation_spearman.png) |
→ 어떤 방식이어도 최대 대략 0.05 정도의 상관 관계를 가진다. → 상관 관계가 거의 없다.
→ 성능 개선이 없음, 속도만 증가
### B. Polynomial Feature 사용
: 비선형 문제를 해결하기 위해 데이터 자체의 차원을 변형시키는 트릭 (예시: 원래 데이터가 $x_1, x_2$일 때, $x_1^2, x_2^2, x_1x_2$ 같은 항을 추가)

>**상위 Feature 선별 후 진행: 그렇지 않을 경우, Feature의 급격한 증가 대비 데이터의 수가 부족**

#### Pearson + Polynomial Degree 2

| Confusion Matrix                           | ROC Curve                           | Learning Curve                           |
| ------------------------------------------ | ----------------------------------- | ---------------------------------------- |
| ![](Figures_MD/confusion_matrix_lr_pearson_poly2.png) | ![](Figures_MD/roc_curve_lr_pearson_poly2.png) | ![](Figures_MD/learning_curve_lr_pearson_poly2.png) |
![](Figures_MD/training_loss_lr_pearson_poly2.png)
- 준수한 성능 (AUC: 0.891)
- 샘플 수 증가에 따라 성능 증가 경향

#### Kendall + Polynomial Degree 2

| Confusion Matrix                           | ROC Curve                           | Learning Curve                           |
| ------------------------------------------ | ----------------------------------- | ---------------------------------------- |
| ![](Figures_MD/confusion_matrix_lr_kendall_poly2.png) | ![](Figures_MD/roc_curve_lr_kendall_poly2.png) | ![](Figures_MD/learning_curve_lr_kendall_poly2.png) |
![](Figures_MD/training_loss_lr_kendall_poly2.png)
- 준수한 성능 (AUC: 0.873)
- 샘플 수 증가에 따라 성능 증가 경향
#### Spearman + Polynomial Degree 2

| Confusion Matrix                            | ROC Curve                            | Learning Curve                            |
| ------------------------------------------- | ------------------------------------ | ----------------------------------------- |
| ![](Figures_MD/confusion_matrix_lr_spearman_poly2.png) | ![](Figures_MD/roc_curve_lr_spearman_poly2.png) | ![](Figures_MD/learning_curve_lr_spearman_poly2.png) |
![](Figures_MD/training_loss_lr_spearman_poly2.png)
- 준수한 성능 (AUC: 0.873)
- 샘플 수 증가에 따라 성능 증가 경향
### L1 Regularization
: 전체 Feature 중 중요도가 낮은 Feature의 가중치를 0으로 변경
(L2의 경우에는 중요도가 낮은 Feature의 가중치를 작게 만듦)

- Solver 변경: ``` lbfgs → saga ```
- Regularization 변경: ``` L2 → L1(Strength: 1.0) ```

| Confusion Matrix                            | ROC Curve                            | Learning Curve                            |
| ------------------------------------------- | ------------------------------------ | ----------------------------------------- |
| ![](Figures_MD/confusion_matrix_lr_regularization.png) | ![](Figures_MD/roc_curve_lr_regularization.png) | ![](Figures_MD/learning_curve_lr_regularization.png) |
![](Figures_MD/training_loss_lr_regularization.png)
- 56%의 가중치가 억제된 것 확인
- 개선된 성능 (AUC: 0.903) → Regularization 적용 전 대비 0.012 증가
- 샘플 수 증가에 따라 성능 증가 경향
## 2. RNN(LSTM) / CNN
### A. 데이터 시계열 특성 유지 (Windowing)

```
기존 입력 shape: (2880, 500, 1)
                  │     │   │
                  │     │   └─ 각 timestep의 feature 1개
                  │     └─ "timesteps" — LSTM이 unroll하는 축
                  └─ batch

변경 후(Rolling Window) shape: (Window_Num, Window_Width, Sensor_data: 500)
```

#### RNN(LSTM)

==**Window Size를 1로 했을 때, AUC 성능이 가장 좋았음
(사실 상 시계열 특성이 없는 것과 마찬가지)**==

| Confusion Matrix                        | ROC Curve                        | Learning Curve                        |
| --------------------------------------- | -------------------------------- | ------------------------------------- |
| ![](Figures_MD/confusion_matrix_rnn_windowing.png) | ![](Figures_MD/roc_curve_rnn_windowing.png) | ![](Figures_MD/learning_curve_rnn_windowing.png) |
![](Figures_MD/training_loss_rnn_windowing.png)
- 개선된 성능 (AUC: 0.826) → Windowing 적용 전 대비 0.041 증가
- 샘플 수 증가에 따라 성능 증가 경향
- **다만 상기 언급한 바와 같이, Window Size가 1이므로 시계열 특성이 활용된 것은 아니며, LSTM의 입력 형태가 다른 것**

>**시계열 특성이 반영되지 않았음에도 LSTM의 성능이 개선된 이유**
>
>**A. No Windowing**
>	500 Time Step에 각각 1개 Feature 입력, Recurrent Weight가 500step을 거치며 학습
>**B. Windowing (Window Size: 1)**
>	1 Time Step에 500개 Feature 입력, Recurrent Weight가 활용되지 않음 (Time Step이 1이라 전파할 곳이 없음)
>	→ **사실 상 MLP**

#### CNN
: Windowing을 적용했을 때, 성능 감소

→ 시계열 특성이 없는데, 계속해서 시계열 특성에 대해 학습하는 것이 오히려 도움이 안됨.

# 모델 성능 요약

| 모델                                      | AUC       | 비고                |
| --------------------------------------- | --------- | ----------------- |
| Logistic Regression                     | 0.476     | 베이스라인, 선형 한계      |
| Logistic Regression + Pearson + Poly2   | 0.891     | 비선형 변환 효과         |
| Logistic Regression + Kendall + Poly2   | 0.873     |                   |
| Logistic Regression + Spearman + Poly2  | 0.873     |                   |
| Logistic Regression + L1 Regularization | 0.903     | 56% 가중치 억제        |
| XGBoost                                 | 0.852     | 파인튜닝 없음           |
| RNN (LSTM)                              | 0.785     | 시계열 입력            |
| RNN (LSTM) + Windowing (W=1)            | 0.826     | 사실상 MLP           |
| **CNN**                                 | **0.970** | **최고 성능**         |

**핵심 인사이트**
- CNN > XGBoost > LR(L1+Poly2) > LSTM > LR 순
- 단순 모델(LR)도 Feature Engineering·정규화로 0.476 → 0.903까지 개선 가능
- 시계열 형태 입력이 항상 LSTM에 유리한 것은 아님 (Window=1일 때 오히려 최고 성능)
- CNN의 우수성은 파인튜닝되지 않은 XGBoost와의 비교라는 점 유의

# 결론

- CNN이 가장 높은 성능을 보여줬으나, XGBoost의 파라미터를 파인튜닝하지 않았음을 고려해야 함. (XGBoost를 파인튜닝 했을 때, 더 높은 성능이 나올 수도 있음)
- 연속적인 센서 데이터라고 해서 꼭 RNN을 사용할 필요는 없다.
	- 어떤 모델을 사용할지 빠르게 선별할 수 있는 것이 큰 경쟁력이 될 것.
- 같은 모델이라도 어떻게 사용하느냐에 따라 결과가 크게 달라진다.
	- 파라미터 파인튜닝의 중요성
- 엔진에서 획득된 500개의 센서 데이터로 실시간 엔진 불량 여부 판정 가능
	- 예측이 아닌, 실시간 분류라면 사실 상 문제가 터진 시점에 문제가 터졌다고 알려주는 것 → 어떤 의미를 가지는가?
		1. 불량 확인 시, 즉각적 분류 및 조치
		2. 24시간 무인 모니터링
		3. 어떤 종류의 이상인지 추정

# Future Work

-  불량 발생 이전에 예측하는 것이 더 큰 가치
	1. 앞으로 N 스텝 안에 불량 발생 가능성이 있음을 판단
	2. 다음 불량까지의 타입 스텝 예측

- 정상 데이터만 가지고, 불량 데이터를 예측

# Reference

- 가이드 북: [[Guidebook_FordEngine.pdf]]
- 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), Ford 엔진 진동 AI 데이터셋, KAIST(AI 대학원 최재식교수), 2020.12.14., www.kamp-ai.kr