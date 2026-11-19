"""stream() for Kitten TTS 0.8 and Kitten TTS 2 mini/nano/pico, the same on every backend.

The chunks come from the native chunker in the engine module (engines/lite/engine/src/stream/chunker.hpp), the same
code the phone library's kt08_stream runs: with chunk_chars=N, whole sentences packed into chunks of at most N
characters; with chunk_chars=None (the default), the model's default: one sentence per chunk for Kitten TTS 0.8 (as
its chunk_text) and 380 characters for Kitten TTS 2 mini/nano/pico (as KittenTTS 2's chunk_chars). Nothing is
measured, so the same text and chunk_chars give the same chunks on every machine. Each chunk is phonemised only when it is next
and synthesised on the model's own backend (CPU, Metal, CUDA).
"""


def check_trim_lead(trim_lead):
    """stream()'s trim_lead: True (default) or False."""
    if not isinstance(trim_lead, bool):
        raise TypeError(f"trim_lead must be True or False, not {trim_lead!r}")
    return trim_lead


def run(chunks, phonemize, synth):
    """Yields the audio of each chunk, in order.

    chunks: the text chunks (engine.stream.pack_sentences). phonemize(text) -> the chunk's phoneme pieces (a list; more
    than one when a Kitten TTS 2 chunk passes the model's 350-phoneme cap; [] for a chunk with nothing to say); it is
    called only when that chunk is next. synth(phonemes, text, first, last) -> the piece's audio (first: the stream's
    first piece; last: no piece follows). The same order of calls as kt08_stream_next in the C library.
    """
    first = True
    for i, text in enumerate(chunks):
        pieces = phonemize(text)
        for j, ph in enumerate(pieces):
            yield synth(ph, text, first, i == len(chunks) - 1 and j == len(pieces) - 1)
            first = False
