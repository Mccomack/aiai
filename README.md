# 음악 분류 모델 (파인튜닝)

/raw: 전처리 전 음악 파일들. 폴더 이름: 라벨, 파일: 음악

ex: /raw/gugak/~~~.wav

/train: 10초 단위로 자르는 전처리를 끝낸 음악 파일들. 라벨링은 위와 같이 함. 

ex: /train/gugak/~~~_chunk001.wav

/test: 실전 테스트. 이것도 나중에 자동화 할 생각 있어서 라벨링은 했지만 없어도 문제 없음. 

ex: /test/gugak/00001.wav

## 실행 방법

일단 uv로 페키지 정보 동기화부터
```bash
uv sync
```

### 모델 학습

허깅페이스 로그인 안 되어 있으면 하는거 권장. 

/raw에 (음악 종류)/(음악 파일) 포멧으로 .wav나 .mp3을 올리고, 아래 명령 실행. 
```bash
uv run python preprocess.py
uv run python train_mode.py
```

### 예측

이미 학습된 모델은 [여기](https://huggingface.co/mccomack/ast-korean-music-genre-classifier)에 있음. 용량 관계상 hf에 올림. 


/test(이 폴더 아니여도 되지만 권장)에 위와 똑같이 포멧해서 음악 파일 올린 후, 아래 명령 실행. 
```bash
uv run python predict.py ./test/(파일 종류)/(음악 파일)
```
