# import gradio as gr
import pandas as pd
import hopsworks
# import matplotlib.pyplot as plt


project = hopsworks.login()
dataset_api = project.get_dataset_api()

data = dataset_api.download("Resources/data/aq_test.csv", None, overwrite = True)
data = pd.read_csv(data)
column1 = "Date"
column2 = "Prediction"


import math

import pandas as pd

import gradio as gr
import datetime
import numpy as np

def get_time():
    return datetime.datetime.now()


plot_end = 2 * math.pi


def get_plot(period=1):
    global plot_end
    data = dataset_api.download("Resources/data/aq_test.csv", None, overwrite = True)
    data = pd.read_csv(data)
    x = data["Date"]
    y = data["Prediction"]
    update = gr.LinePlot.update(
        value=pd.DataFrame({"Date": x, "AQI Prediction": y}),
        x="Date",
        y="AQI Prediction",
        title="Beijing Air Quality Index Forecast",
        width=600,
        height=350,
    )
    plot_end += 2 * math.pi
    if plot_end > 1000:
        plot_end = 2 * math.pi
    return update

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Textbox(
                "Beijing Air Quality Index Forecast For the Next 7 Days",
                label="",
            )
            plot = gr.LinePlot(show_label=False)


    dep = demo.load(get_plot, None, plot, every=1)
    plot.change(get_plot, plot, every=1, cancels=[dep])

if __name__ == "__main__":
    demo.queue().launch()