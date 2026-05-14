# Ford Engine Fault Detection

Ford 엔진 진동 데이터셋으로 다양한 분류 모델(LR, XGBoost, RNN, CNN)을 비교하고 성능 개선 기법을 적용한 실험 프로젝트.

## 데이터셋

- **출처**: KAMP (Korea AI Manufacturing Platform), Ford 엔진 진동 AI 데이터셋 (2020.12, KAIST 최재식 교수)
- **구조**: 500 sensor × 시계열 timestep
- **라벨**: 시점별 정상(1) / 비정상(0) 이진 분류

## 환경 설정

```bash
# Conda 환경 생성
conda create -n KAMP_FordEngine python=3.10
conda activate KAMP_FordEngine

# 패키지 설치
pip install numpy pandas scipy scikit-learn matplotlib xgboost
pip install "tensorflow[and-cuda]"   # GPU 사용 시
```

## 사용법

`src/main.py`에서 `MODEL` 변수 하나만 바꿔서 모델 전환:

```python
MODEL = "LR"   # "LR" | "XGB" | "RNN" | "CNN"
```

실행:
```bash
cd src
python main.py
```

각 모델에는 별도 하이퍼파라미터 (`LR_TOP_K`, `LR_PENALTY`, `LR_C`, `USE_WINDOWING`, `WINDOW_SIZE` 등)가 있어 같은 main.py 안에서 조절 가능.

## 프로젝트 구조

```
.
├── src/
│   ├── main.py                    # 전체 파이프라인 (모델 선택 + 학습 + 평가)
│   ├── EnvTest.py                 # 환경/라이브러리 점검
│   ├── LoadData_FordEngine.py     # ARFF 데이터 로딩
│   ├── Split.py                   # train/valid 분할
│   ├── Correlation.py             # 상관관계 계산 유틸
│   ├── Windowing.py               # rolling window 변환
│   ├── Visualization.py           # 모든 시각화 함수
│   └── Models/
│       ├── LogisticRegression.py
│       ├── XGBoost.py
│       ├── RNN.py
│       └── CNN.py
├── dataset/                       # FordA ARFF 데이터 (KAMP 홈페이지에서 다운로드 가능)
├── models/                        # 학습된 모델 체크포인트 (gitignored)
└── document/
    └── KAMP_FordEngine.md         # 상세 실험 노트
```

## Baseline 성능 비교

| 모델 | 종류 | AUC | 비고 |
|---|---|---|---|
| Logistic Regression | Linear | 0.476 | 선형 분류기의 한계 |
| XGBoost | Ensemble (Tree) | 0.852 | tabular 비선형 강력 |
| RNN (LSTM) | Neural Network | 0.785 | 시계열 모델 |
| **CNN** | Neural Network | **0.970** | 가장 우수 |

## 성능 개선 실험 요약

### Logistic Regression
| 기법 | AUC |
|---|---|
| Baseline | 0.476 |
| + Top-K Feature Selection | 변화 없음 (상관계수 자체가 약함) |
| + Polynomial Feature (degree=2) | 0.891 |
| + L1 Regularization (Polynomial 위에) | **0.903** |

→ 약 56%의 가중치가 L1으로 억제됨. 선형 → 비선형 + Regularization으로 baseline 대비 대폭 개선.

### RNN / CNN
| 기법 | RNN AUC | CNN AUC |
|---|---|---|
| 기본 입력 `(N, 500, 1)` | 0.785 | 0.970 |
| Windowing (W=1) | 0.826 | 감소 |

- RNN: Window=1이 best → 사실상 MLP 동작 → **이 데이터에 시계열 정보가 거의 없음**을 시사
- CNN: Windowing 적용 시 오히려 성능 감소 → 마찬가지 시사

## 주요 발견

1. **데이터의 본질은 tabular**
   - 모든 sensor의 |Pearson corr with target| ≈ 0.05 → 선형 신호 약함
   - RNN의 best window size = 1 → 시계열 정보 거의 없음
   - "한 시점의 500 sensor 조합 → 분류"가 본질

2. **모델 선택의 직관 ≠ 결과**
   - "연속적 sensor 데이터 = RNN" 이라는 직관이 항상 맞지 않음
   - CNN이 가장 좋고, RNN은 평범. XGBoost도 baseline 상태에서 0.85
   - 빠른 EDA + cheap baseline 비교로 후보 좁히는 능력이 핵심

3. **같은 모델도 사용 방식에 따라 결과 큼**
   - LR baseline (0.476) → LR + Polynomial + L1 (0.903): 거의 두 배
   - Window size, regularization 등 작은 변수가 큰 차이

## 비즈니스 가치 — 한계와 가능성

현재 모델은 **실시간 이상 감지(detection)**이지 **사전 예측(prediction)**은 아님.

**가치 있는 use case**:
- 실시간 자동 정지 (안전)
- 24/7 무인 모니터링
- 조립 라인 품질 검사
- 정비 진단 보조

**Future Work** (더 큰 가치):
- N step 후 고장 가능성 예측 (predictive maintenance)
- Remaining Useful Life (RUL) 회귀
- 정상 데이터만으로 이상 예측 (one-class anomaly detection)

## 상세 문서

전체 실험 과정과 시각화는 [document/KAMP_FordEngine.md](document/KAMP_FordEngine.md) 참고.

## Reference

- 중소벤처기업부 KAMP, *Ford 엔진 진동 AI 데이터셋*, KAIST AI 대학원 최재식 교수, 2020.12.14
- [www.kamp-ai.kr](https://www.kamp-ai.kr)
