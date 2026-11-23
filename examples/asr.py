from kitten_inference import KittenASR

# Models: KittenML/kitten-asr-tiny (0.3B), KittenML/kitten-asr-small-enhanced (0.6B).
# Runs on an NVIDIA GPU when PyTorch is installed and finds one (pip install torch), otherwise on the CPU
# (Linux x86-64 and arm64, Macs with Apple silicon; no PyTorch needed). Python 3.10 or newer.
# backend="cuda", "cpu" or "metal" picks one; precision picks the number format of the weights.
asr = KittenASR("KittenML/kitten-asr-tiny")
print(asr.capabilities["backend"], asr.transcribe("meeting.wav"))

# A list of files is transcribed in one call; the result is one text per file.
for text in asr.transcribe(["a.wav", "b.wav"]):
    print(text)

# details=True returns a Transcript: text, language, segments and raw.
result = asr.transcribe("meeting.wav", language="English", details=True)
print(result.language, result.text)

# Live audio: push 16 kHz mono samples as they arrive.
# with asr.stream() as live:
#     for chunk in microphone_chunks:
#         update = live.push(chunk)
#         if update is not None:
#             print(update.text)
#     print(live.finish())
asr.close()
