#  Predicting Heart Disease Rate of College Students<br>

### 사용한 데이터셋

<h3>Kaggle - Personal Key Indicators of Heart Disease</h3>
→ https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease <br><br>

총 18개의 컬럼으로 구성되어 있습니다. <br>

- HeartDiseae : 심장병(CHD or MI) 여부 <br>
- BMI : BMI 수치 <br>
- Smoking : 살면서 최소 100개의 담배를 피운 적 있는지 <br>
- AlcoholDrinking : 음주량이 많은지 (성인 남성 - 주당 14잔 이상, 성인 여성 - 주당 7잔 이상) <br>
- Stroke : Stroke이 온 적이 한 번이라도 있는지<br>
- PhysicalHealth : 신체적 질병과 부상을 포함해서, 지난 30일 동안 신체적 건강이 좋지 않았던 날의 수 (0~30) <br>
- MentalHealth : 지난 30일 동안 정신 건강이 좋지 않았던 날의 수 (0~30) <br>
- DiffWalking : 걷거나 계단을 오르는 데 심각한 어려움을 겪는지 <br>
- Sex : 남성 or 여성 <br>
- AgeCategory : 14개의 연령대 (n0-n4, n5-n9) <br>
- Race : 인종 (백인, 히스패닉, 흑인, 아시아인 등) <br>
- Diabetic : 당뇨 여부 <br>
- PhysicalActivity : 정규 직장을 제외하고 30일 동안 신체 활동이나 운동을 해왔는지 <br>
- GenHealth : 전반적인 건강 상태 (Excellent, Very Good, Good, Fair, Poor) <br>
- SleepTime : 수면 시간 <br>
- Asthma : 천식 여부 <br>
- KidneyDisease : 신장 결석, 방광염, 요실금을 제외한 신장 질환을 겪은 적 있는지 <br>
- SkinCancer : 피부암 여부 <br>
<br><br>

### Manual

위 파일을 다운로드하여 Colab Notebooks에서 엽니다. 파일 내 적혀있는 스크립트를 기반으로 위에서부터 순차적으로 셀 실행을 합니다. <br>
<br><br>

### 개발 환경

Google Colab Notebooks. 아래 코드를 사용하여 환경 및 세팅을 확인합니다. <br>
>CPU information

```python
!head /proc/cpuinfo
```
<br>

>Memeory information

```python
!head -n 3 /proc/meminfo
```
<br><br>

### Data import 

캐글에 있는 데이터를 사용하기 위해서 캐글 api 파일을 업로드합니다. <br> 
> 깃허브 첨부파일 사용<br>

```python
!pip install kaggle
from google.colab import files
files.upload()
```

위 코드 실행 후 파일 선택 창에서 **kaggle.json** 파일을 업로드합니다. <br>
<br><br>

> 개인 캐글 API 사용<br>

Kaggle에 로그인한 후, **My Account**로 들어가서 스크롤을 아래로 내리면 보이는 **Create New API Token** 버튼을 클릭하여 **kaggle.json** 파일을 다운받습니다. <br>

동일 코드 실행 후 파일 선택 창에서 본인의 **kaggle.json** 파일을 업로드합니다. <br>
<br><br>

### Data Preprocessing

Train : Validation : Test 비율 - 60 : 20 : 20 <br>
Feature간 Correlation을 확인한 결과 강한 상관을 보이는 것은 없으므로 모든 features를 선택하여 진행합니다. 레이블 인코딩 방식을 사용했습니다. 
<br>

train data - `X_train, y_train` <br>
validation data - `X_val, y_val` <br>
test data - `X_test, y_test` <br>
<br><br>

### Imbalancing Problem

SMOTE 기법을 활용하여 오버샘플링을 합니다. <br>

```python
from imblearn.over_sampling import SMOTE
over = SMOTE(k_neighbors=2000)
X_train_balanced,y_train_balanced = over.fit_resample(X_train, y_train)
```
train data를 오버샘플링 하였기 때문에 `X_train → X_train_balanced`, `y_train → y_train_balanced`가 됩니다. 

<br><br>

### Model

사용한 모델은 다음 네 개입니다. LR과 RF는 **SMOTE** 기법을 처리한 데이터와 하지 않은 데이터 두 개 모두 확인할 수 있습니다. 

> Logistic Regression 
  - w/o SMOTE<br>
  `Standard Scaler`를 사용하여 데이터를 정규화하고, 그리드 서치(비교 차원에서 기준은 Accuracy, F1 score 둘 다 사용)를 통해 하이퍼 파라미터 튜닝을 진행합니다. <br>
  나온 결과로 **Best Model**을 구해 Validation Set을 사용하여 Accuracy, Precision, Recall, F1 score를 출력하고 Confusion Matrix를 확인합니다. <br><br>


  - w/ SMOTE <br>
  밸런스를 맞추지 않은 데이터와 마찬가지로 `Standard Scaler`를 사용하여 데이터를 정규화하고, 그리드 서치(비교 차원에서 기준은 Accuracy, F1 score 둘 다 사용)를 통해 하이퍼 파라미터 튜닝을 진행합니다. <br>
  나온 결과로 **Best Model**을 구해 Validation Set을 사용하여 Accuracy, F1 score를 출력하고 Confusion Matrix를 확인합니다. <br>
  <br><br>


> Random Forest 
  - w/o SMOTE <br>
  Default값으로 모델을 생성한 후 이를 교차 검증하여 평가합니다. 그리드 서치에 시간 소요가 커 랜덤 서치(기준 F1 score)를 통해 하이퍼 파라미터 튜닝을 진행하고, 이를 기반으로 다시 훈련하여 그 시간을 측정합니다. <br>
  나온 결과로 **Best Model**을 구해 Validation Set을 사용하여 Accuracy, Precision, Recall, F1 score를 출력하고 Confusion Matrix를 확인합니다. <br><br>

  - w/ SMOTE <br>
  Default값으로 모델을 생성한 후 Validation Set을 사용하여 평가합니다. 그리고 위에서 튜닝한 하이퍼 파라미터와 밸런스를 맞춘 데이터로 훈련 후 똑같이 Validation을 진행합니다. <br>
  튜닝된 모델이 디폴트 파라미터를 사용한 모델보다 성능이 나음을 볼 수 있습니다. 마찬가지로 Accuracy, Precision, Recall, F1 score를 출력하고 Confusion Matrix를 확인합니다.<br>
  <br><br>

> GBM 
  - w/o SMOTE <br>
  Default값으로 모델을 생성한 후 이를 교차 검증하여 평가합니다. 그리드 서치(기준 - Accuracy, F1)를 통해 하이퍼 파라미터 튜닝을 진행하고, 이를 기반으로 다시 훈련합니다. <br>
  나온 결과로 **Best Model**을 구해 Validation Set을 사용하여 Accuracy, Precision, Recall, F1 score를 출력하고 Confusion Matrix를 확인합니다. <br>
  <br><br>

  - w/ SMOTE <br>
  Default값으로 모델을 생성한 후 Validation Set을 사용하여 평가합니다. 그리고 위에서 튜닝한 하이퍼 파라미터와 밸런스를 맞춘 데이터로 훈련 후 똑같이 Validation을 진행합니다. <br>
  튜닝된 모델이 디폴트 파라미터를 사용한 모델보다 성능이 나음을 볼 수 있습니다. 마찬가지로 Accuracy, Precision, Recall, F1 score를 출력하고 Confusion Matrix를 확인합니다.<br>
  <br><br>


> XBG<br>

다음과 같은 파라미터를 사용하여 모델을 훈련하고 검증합니다. <br> `max_depth = 20, n_estimators = 250, early_stopping_rounds = 50, eval_beric = 'auc', eval_set = [(X_val, y_val)]` <br>
그리드 서치에 드는 시간을 단축시키기 위해 트리를 축소하고, 조기 중단값 역시 `30`으로 줄입니다. cv 값 역시 4(혹은 3) 정도로 조절합니다. 기준은 똑같이 Accuracy와 F1 score를 사용합니다. 그리드 서치를 통해 하이퍼 파라미터 튜닝을 진행합니다.<br>
나온 결과로 **Best Model**을 구해 Validation Set을 사용하여 Accuracy, Precision, Recall, F1 score, 그리고 Confusion Matrix를 확인합니다. <br>
<br><br>

### Model Assessment

Random Forest 모델 평가를 위해 여러 가지 기법을 사용하고 시각화를 진행하였습니다. 아래 내용을 확인할 수 있습니다. <br><br>
- Baseline Models <br>
- Confusion Matrix <br>
- Precision, Recall, F1 score, Support <br>
- Predicted Probability of Each Point <br>
- ROC-AUC <br>
- Tree Visualization <br>
<br><br>

### Interpretation of the result

Feature Importance를 확인합니다. <br>
<br><br>