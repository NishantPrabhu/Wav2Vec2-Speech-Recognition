
import os
import time
import dash
import wave
import models
import pyaudio
import librosa
import soundfile
import numpy as np
from datetime import datetime as dt
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input 

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# ======================================
# App layout
# ======================================

app.layout = html.Div(children=[

    html.H1(
        "Automatic Speech Recognition",
        style = {"width": "100%", "padding-top": "200px", "text-align": "center", "font-weight": "bold", "padding-left": "100px", "padding-right": "100px"}
    ), 

    html.Div(
        style = {"width": "100%", "float": "left", "padding-left": "200px", "text-align": "center", "padding-right": "200px", "padding-top": "50px"},
        children = [
            html.Button(
                children=["Toggle recording"], 
                style={"background-color": "#2295d4", "color": "white", "height": "50px", "width": "250px", "border": {"width": "0px"}},
                id="record-button", 
                n_clicks=0
            ),
            html.Div(id="hidden-div", style={"display": "none"}),
            html.Div(id="app-status", style={"padding-top": "20px", "width": "100%", "text-align": "center"}),
            html.Div(id="record-status", style={"padding-top": "20px", "width": "100%", "text-align": "center"}),
            html.H4("Prediction", style={"padding-top": "40px", "width": "100%", "text-align": "center", "font-weight": "bold"}),
            html.Div(id="prediction-container", style={"padding-top": "20px", "width": "100%", "text-align": "center"})
        ]
    )
])

# ======================================
# App Callbacks
# ======================================

# Global stuff hopefully
frames = []
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024)
curr_time = dt.now().strftime("%d-%m-%Y_%H-%M")

@app.callback(
    [Output("record-button", "children"), Output("record-button", "style"), Output("app-status", "children")],
    [Input("record-button", "n_clicks")]
)
def start_or_stop_recording(n_clicks):
    stopped_style = {"background-color": "#2295d4", "color": "white", "height": "50px", "width": "150px"}
    started_style = {"background-color": "#bd263a", "color": "white", "height": "50px", "width": "150px"}
    if (n_clicks % 2) != 0:
        return ["Stop recording"], started_style, "Recording in progress..."
    else:
        return ["Start recording"], stopped_style, "Press the button above to start recording."


@app.callback(
    Output("record-status", "children"),
    [Input("record-button", "n_clicks")]
)
def record_audio(n_clicks): 
    if (n_clicks % 2) != 0:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024)
        for i in range(int(44100 / 1024 * 5)):
            data = stream.read(1024)
            frames.append(data)
            if (n_clicks % 2) == 0:
                break 

@app.callback(
    Output("hidden-div", "children"),
    [Input("record-button", "n_clicks")]
)
def stop_recording(n_clicks):
    if (n_clicks % 2) == 0:
        stream.stop_stream()
        stream.close()
        p.terminate()
        with wave.open(f"test_recording_{curr_time}.wav", "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b"".join(frames))
        return ["None"]

@app.callback(
    Output("prediction-container", "children"),
    [Input("record-button", "n_clicks")]
)
def generate_prediction(n_clicks):
    if os.path.exists(f"test_recording_{curr_time}.wav") and (n_clicks % 2) == 0:
        start_time = time.time()
        pred_str = model.predict_for_file(f"test_recording_{curr_time}.wav")
        print("Prediction time: {} sec".format(time.time() - start_time))
        os.remove(f"test_recording_{curr_time}.wav")
        return [pred_str[0][0] + pred_str[0][1:].lower()]
    return ["Please generate a recording!"]

# ======================================
# Main
# ======================================

if __name__ == "__main__":

    # Initialize the model
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

    app.run_server(debug=True)
