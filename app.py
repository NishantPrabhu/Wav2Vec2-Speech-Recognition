
import os
import time
import dash
import wave
import models
import pyaudio
import librosa
import soundfile
import numpy as np
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input 
from datetime import datetime as dt

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
            html.Div(style={"width": "50%", "display": "inline-block", "padding-left": "300px"}, children=[
                html.Button(
                    children=["Start recording"], 
                    style={"background-color": "#2295d4", "color": "white", "height": "50px", "width": "250px", "border": {"width": "0px"}},
                    id="start-button", 
                    n_clicks=0
                ),
            ]),
            html.Div(style={"width": "50%", "display": "inline-block", "padding-right": "300px"}, children=[
                html.Button(
                    children=["Stop recording"], 
                    style={"background-color": "#bd263a", "color": "white", "height": "50px", "width": "250px", "border": {"width": "0px"}},
                    id="stop-button", 
                    n_clicks=0
                ),
            ]),
            html.Div(id="hidden-div", style={"display": "none"}),
            html.Div(id="hidden-div-2", style={"display": "none"}),
            html.Div(id="app-status", style={"padding-top": "40px", "width": "100%", "text-align": "center"}),

            html.H4("Prediction", style={"padding-top": "40px", "width": "100%", "text-align": "center", "font-weight": "bold"}),
            html.Div(id="prediction-container", style={"padding-top": "20px", "width": "100%", "text-align": "center"})
        ]
    )
])

# ======================================
# App Callbacks
# ======================================

curr_time = dt.now().strftime("%d-%m-%Y_%H-%M")
stopped = False

@app.callback(
    Output("hidden-div", "children"),
    [Input("start-button", "n_clicks")]
)
def record_audio(start_clicks):
    stopped = False
    if start_clicks > 0:
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, frames_per_buffer=1024) 
        for i in range(int(44100 / 1024 * 10)):
            data = stream.read(1024)
            frames.append(data)
            if stopped:
                break
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
    [Output("hidden-div-2", "children"), Output("app-status", "children")],
    [Input("start-button", "n_clicks"), Input("stop-button", "n_clicks")]
)
def stop_recording(start_clicks, stop_clicks):
    if (stop_clicks > 0):
        if stop_clicks == start_clicks:
            stopped = True           
            return ["None"], ["Please wait while your recorded audio is being processed."]
        else:
            return ["None"], ["Start recording by clicking the 'Start recording' button"]
    else:
        return ["None"], ["Start recording by clicking the 'Start recording' button"]

@app.callback(
    Output("prediction-container", "children"),
    [Input("hidden-div", "children")]
)
def generate_predictions(dummy):
    while not os.path.exists(f"test_recording_{curr_time}.wav"):
        continue
    start_time = time.time()
    pred_str = model.predict_for_file(f"test_recording_{curr_time}.wav")
    print("Prediction time: {} sec".format(time.time() - start_time))
    os.remove(f"test_recording_{curr_time}.wav")
    return [pred_str[0][0] + pred_str[0][1:].lower()]
    
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
