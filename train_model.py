import os

# Hugging Face 데이터셋 캐시가 저장될 위치 명시적으로 지정
os.environ["HF_DATASETS_CACHE"] = "./local_cache"

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)

def main():
    # 가속기(MPS) 있으면 활용
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # train 안의 데이터셋 불러오고
    dataset = load_dataset("audiofolder", data_dir="train", drop_labels=False)
    
    # 모델 입력 규격(16kHz)으로 오디오 포맷 맞추기
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # 학습(Train)과 검증(Validation) 세트로 8:2 분할 (과적합 방지) 
    splits = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = splits["train"]
    val_ds = splits["test"]

    # 장르 정보 추출
    labels = train_ds.features["label"].names
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    print(f"found geners: {labels}")

    # 베이스 모델 불러오기
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(
        model_id, 
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True # 기존 서양 분류 클래스를 지우고 한국 음악 장르로 덮어씌움
    )
    model.to(device)

    # 스펙토그램으로 변환
    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays, 
            sampling_rate=feature_extractor.sampling_rate, 
            max_length=160000, # 10초 * 16kHz
            truncation=True
        )
        return inputs

    # 데이터셋에 전처리 일괄 적용
    encoded_train = train_ds.map(preprocess_function, remove_columns="audio", batched=True)
    encoded_val = val_ds.map(preprocess_function, remove_columns="audio", batched=True)

    # 평가 지표 설정
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # 학습 정보? 설정
    training_args = TrainingArguments(
        output_dir="./genre_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2, # 한번에 학습 데이터 적게
        gradient_accumulation_steps=8, # 학습 횟수 많게
        per_device_eval_batch_size=2,
        num_train_epochs=5, # 전체 데이터를 5번 반복 학습
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        # use_mps_device=True, # 최신에는 자동으로 MPS 여부 감지해서 킴
        push_to_hub=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_val,
        processing_class=feature_extractor,
        compute_metrics=compute_metrics,
    )

    # 제발 커널 패닉만은 안 된다
    trainer.train()

    # 모델 저장
    trainer.save_model("./final_genre_model")
    print("저장 위치: ./final_genre_model")

if __name__ == "__main__":
    main()