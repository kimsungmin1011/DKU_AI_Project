# 단국대학교 인공지능 수업 실습


## Electra 모델을 활용한 금융 및 법률 머신러닝

이 프로젝트는 Electra 모델을 사용하여 금융 및 법률 분야에서의 머신러닝 응용에 중점을 두었습니다. 
- OS: Windows 11, macOS Ventura 13.0.1
- Python: 3.11.1
- 데이터 전처리, 모델 학습, 모델 검증 과정은 다음과 같습니다.

### 제작
- 김성민 (컴퓨터공학과)

### 목차
1. [Electra 모델 개요](#electra-모델-개요)
2. [데이터 전처리](#데이터-전처리)
3. [설치 방법](#설치-방법)
4. [모델 학습](#모델-학습)
5. [모델 검증](#모델-검증)

---

### Electra 모델 개요
Electra 모델은 자연어 처리 작업에 효과적인 최신 변형기반(transformer-based) 모델입니다. BERT와 같은 다른 모델들과 달리 사전 훈련 방식에서 독특한 접근 방식을 취합니다.

### 데이터 전처리
<img width="672" alt="스크린샷 2023-12-22 오후 8 24 13" src="https://github.com/kimsungmin1011/Open-source-AI-Project/assets/122242600/580ac15c-2f3e-48a1-8931-98746d6d5519">

**파일:** `preprocessing.py`

이 스크립트는 훈련을 위한 데이터 준비를 담당합니다. 데이터를 정리하고, 정규화하며, Electra 모델에 적합한 형식으로 구조화하는 작업을 포함합니다.

### 설치 방법
이 프로젝트에 필요한 환경을 설정하려면 다음 명령어를 실행하세요:

```bash
pip install accelerate -U
pip install transformers[torch]
pip install datasets
```
### 모델 학습
<img width="1127" alt="스크린샷 2023-12-22 오후 8 17 05" src="https://github.com/kimsungmin1011/-AI-/assets/122242600/8a1d6768-a702-4a75-b305-1cc7d6dd8abf">

**파일:** `training.py`

이 스크립트는 전처리된 데이터로 Electra 모델을 훈련하는 과정을 설명합니다. 모델 설정, 훈련 매개변수 정의, 훈련 프로세스 시작 등을 포함합니다.

### 모델 검증
<img width="1168" alt="스크린샷 2023-12-22 오후 8 18 40" src="https://github.com/kimsungmin1011/-AI-/assets/122242600/fd28f4e0-9a05-41b7-b59d-2b74471cfc38">

**파일:** `validation.py`

훈련 후, 이 스크립트를 사용하여 모델의 성능을 검증합니다. 모델을 검증 데이터로 테스트하는 기능을 포함하며, 금융 및 법률 맥락에서의 정확성과 효과성을 평가하는 답변을 제공합니다.
