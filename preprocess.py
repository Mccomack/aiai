import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

def preprocess_dataset(raw_dir="raw", train_dir="train", chunk_duration=10, target_sr=16000):
    print(f"preprocessing: '{raw_dir}' -> '{train_dir}'")

    if not os.path.exists(raw_dir):
        print(f"cannot find '{raw_dir}'. ")
        return

    os.makedirs(train_dir, exist_ok=True)

    chunk_samples = chunk_duration * target_sr

    genres = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    if not genres:
        print(f"cannot find labeled folders on '{raw_dir}'. ")
        return

    total_chunks_created = 0

    for genre in genres:
        genre_raw_path = os.path.join(raw_dir, genre)
        genre_train_path = os.path.join(train_dir, genre)
        os.makedirs(genre_train_path, exist_ok=True)

        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.WAV", "*.MP3"]:
            audio_files.extend(Path(genre_raw_path).rglob(ext))

        print(f"\nprocessing [{genre}]... (found items: {len(audio_files)})")

        genre_chunk_count = 0

        for file_path in audio_files:
            file_name = file_path.stem

            try:
                y, sr = librosa.load(file_path, sr=target_sr)

                for i in range(0, len(y), chunk_samples):
                    chunk = y[i : i + chunk_samples]

                    # 10초보다 짧은 마지막 조각이나 짧은 원본 파일은 무음으로 채움
                    if len(chunk) < chunk_samples:
                        # 1초 미만은 넘김 (학습에 방해됨)
                        if len(chunk) < target_sr * 1:
                            continue

                        padding = chunk_samples - len(chunk)
                        chunk = np.pad(chunk, (0, padding), mode='constant')

                    chunk_file_name = f"{file_name}_chunk{i // chunk_samples:03d}.wav"
                    chunk_file_path = os.path.join(genre_train_path, chunk_file_name)
                    
                    sf.write(chunk_file_path, chunk, target_sr)
                    genre_chunk_count += 1
                    total_chunks_created += 1

            except Exception as e:
                print(f"skipped '{file_name}' due to an error: {e}")

        print(f"[{genre}] completed: {genre_chunk_count} total train datas. ")

    print(f"\ncompleted. ")

if __name__ == "__main__":
    preprocess_dataset(raw_dir="raw", train_dir="train", chunk_duration=10, target_sr=16000)