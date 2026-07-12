import argparse
import librosa
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

data_label = ['classical', 'gugak', 'hiphop', 'jazz', 'rock', 'trot'] # 학습할 떄 라벨 그대로

def predict_genre(file_path, model_dir="./final_genre_model"):
    try:
        # 학습된 모델과 특성 추출기(Feature Extractor) 로드
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
        model = AutoModelForAudioClassification.from_pretrained(model_dir)
    except Exception as e:
        print(f"cannot find model. \n{e}")
        return

    # 가속기(MPS)가 있다면 활용
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval() # 추론 모드로 전환

    print(f"reading '{file_path}'...")
    try:
        # 학습 때와 동일하게 16kHz로 맞춤
        audio_array, sr = librosa.load(file_path, sr=feature_extractor.sampling_rate)
    except Exception as e:
        print(f"cannot read audio file. \n{e}")
        return

    # 10초짜리 여러개인 오디오로 자름. (학습할 떄 10초짜리만 함)
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sr,
        max_length=160000, # 10초 분량의 샘플 수
        truncation=True,   # 10초보다 길면 앞부분 10초만 사용
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 모델로 예측
    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
    
    print(" - - - predict result - - - ")

    sorted_indices = np.argsort(probabilities)[::-1]
    
    for idx in sorted_indices:
        label = data_label[idx]
        score = probabilities[idx]
        print(f"[{label}]: {score * 100:.2f}%")

    top_label = data_label[sorted_indices[0]]
    print(f"predicted genre: {top_label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="음악 장르 추측 모델")
    parser.add_argument("file_path", type=str, help="분석할 음악 파일 경로 (.wav 또는 .mp3)")
    
    args = parser.parse_args()
    predict_genre(args.file_path)