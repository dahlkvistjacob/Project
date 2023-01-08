import gradio as gr

demo = gr.Interface(
    fn=titanic,
    title="Titanic survival prediction",
    description="Experiment with parameters to predict if the fictional passenger survived",
    allow_flagging="never",
    inputs=[
        gr.inputs.Dropdown(["1st class", "2nd class", "3rd class"], label="Ticket class"),
        gr.inputs.Dropdown(["female", "male"], label="Sex"),
        gr.inputs.Dropdown(["Cherbourg", "Queenstown", "Southampton"], label="Port of Embarkation"),
        gr.inputs.Number(default=50.0, label="Fare"),
        gr.inputs.Number(default=20.0, label="Age"),
        gr.inputs.Number(default=0, label="Number of siblings/spouses aboard the Titanic"),
        gr.inputs.Number(default=0, label="Number of parents/children aboard the Titanic"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch()