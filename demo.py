import gradio as gr

def greet(name, intensity):
    return "Hello " + name + "!" * int(intensity)

demo = gr.Interface(
    fn = greet,
    inputs = [
        gr.Textbox(label="Name", placeholder="Pepe, Juan..."),
        gr.Slider(minimum=1, maximum=10, step=1, value=2)
    ],
    outputs=gr.Textbox(label="Greeting")
)

demo.launch()