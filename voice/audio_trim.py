"""Cut the silence at both ends of a recorded line (plan amendments 13:31 and 13:50).

The rule: cut the clip into 20 ms frames and take each frame's loudness (root mean square). A frame is
loud if it is louder than 10% of the clip's 95th-percentile frame. Speech is a run of at least 5 loud
frames in a row (100 ms), so a key click is ignored. Keep from 100 ms before the first speech run to
100 ms after the last one.
"""
import numpy as np

RATE = 16000  # 16 kHz mono audio, as recorded and as Whisper expects
FRAME = int(0.020 * RATE)  # 20 ms frames
PAD = int(0.100 * RATE)  # 100 ms kept before the first and after the last speech run
SPEECH_RUN_FRAMES = 5  # speech = at least 5 loud frames in a row; shorter is a click
LITTLE_TRIMMED = 0.9  # flag a clip if more than 90% of it is kept


def trim_silence(audio):
    """Return (trimmed audio, speech_end_s, flag).

    speech_end_s: seconds from the start of the trimmed clip to the end of the last speech frame.
    flag: "" if all is well, otherwise a short note (the clip is then returned untrimmed if no speech).
    """
    n_frames = len(audio) // FRAME
    if n_frames == 0:
        return audio, len(audio) / RATE, "too short to trim"
    loudness = np.sqrt(np.mean(audio[: n_frames * FRAME].reshape(n_frames, FRAME) ** 2, axis=1))
    loud = loudness > 0.1 * np.percentile(loudness, 95)

    # Find the runs of loud frames that are long enough to be speech.
    runs = []  # (first frame, last frame) of each speech run
    run_start = None
    for i, is_loud in enumerate(list(loud) + [False]):  # the extra False closes a run at the end
        if is_loud and run_start is None:
            run_start = i
        if not is_loud and run_start is not None:
            if i - run_start >= SPEECH_RUN_FRAMES:
                runs.append((run_start, i - 1))
            run_start = None
    if not runs:
        return audio, len(audio) / RATE, "no speech run found: untrimmed"

    begin = max(0, runs[0][0] * FRAME - PAD)
    speech_end = (runs[-1][1] + 1) * FRAME  # sample where the last speech frame ends
    end = min(len(audio), speech_end + PAD)
    flag = "little trimmed (over 90% kept)" if (end - begin) / len(audio) > LITTLE_TRIMMED else ""
    return audio[begin:end], (speech_end - begin) / RATE, flag
