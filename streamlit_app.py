
import os
import av 
import time 
import pydub
import queue
import models 
import threading
import numpy as np
import streamlit as st
from collections import deque
from datetime import datetime as dt
from streamlit_webrtc import AudioProcessorBase, ClientSettings, WebRtcMode, webrtc_streamer


def main():

    # Load model
    args = {
        "config": "configs/main.yaml", 
        "output": dt.now().strftime("%d-%m-%Y_%H-%M"), 
        "task": "single_test", 
        "device": "cpu",
        "dataset": "timit", 
        "load": "outputs/timit/train/31-05-2021-11-58",
        "file": "test_recording.wav"
    }
    model = models.Trainer(args)

    # Application
    st.header("Automatic Speech Recognition")
    st.markdown("Wav2Vec 2.0 model finetuned on TIMIT English dataset")

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-hyperverge",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True}
        )
    )

    status_indicator = st.empty()
    if not webrtc_ctx.state.playing:
        return 

    status_indicator.write("Loading...")
    text_output = st.empty()

    while True:
        if webrtc_ctx.audio_receiver:
            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No audio arrived")
                continue

            status_indicator.write("Recording! Say something!")
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels)
                )
                sound_chunks += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
                buffer = np.array(sound_chunk.get_array_of_samples())
                pred_text = model.predict_on_array(buffer)
                text_output.markdown(f"**Text: ** {pred_text}")
        else:
            status_indicator.write("AudioReceiver not set. Abort.")
            break


if __name__ == "__main__":

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]
    main()
