import transformers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import audiocap
import librosa

def main():
    print(f"Trasformer version: {transformers.__version__}")
    audio_dir = Path("data/clotho_v2.1/audiofolder")
    architecture_name = "MU-NLPC/whisper-tiny-audio-captioning"
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture_name, language="en", task="transcribe")
    model = audiocap.WhisperForAudioCaptioning.from_pretrained(architecture_name)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture_name)

    input_file = audio_dir/"woodpecker1.wav"
    audio, sampling_rate = librosa.load(input_file, sr=feature_extractor.sampling_rate)
    features = feature_extractor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # Prepare caption style
    style_prefix = "clotho > caption: "
    style_prefix_tokens = tokenizer("", text_target=style_prefix, return_tensors="pt", add_special_tokens=False).labels

    model.eval()
    outputs = model.generate(
        inputs=features,
        forced_ac_decoder_ids=style_prefix_tokens,
        max_length=100,
    )
    print(f"output: {tokenizer.batch_decode(outputs)}")

if __name__ == "__main__":
    main()
    