from transformers import pipeline


# transcriber = pipeline(model="openai/whisper-large-v2")
#transcriber = pipeline(model="openai/whisper-small", device_map="mps")
transcriber = pipeline(model="openai/whisper-small", device_map="auto")
transcriber.generation_config.language = "en"

out_1 = transcriber(inputs="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(out_1)

out_2 = transcriber(inputs="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/4.flac")
print(out_2)

out_3 = transcriber(inputs="https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(out_3)


