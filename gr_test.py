#!/ext4/pyenv/llm/bin/python

import gradio as gr
import time

def example1():
    def greet(name):
        return "Hello " + name + "!"


    with gr.Blocks() as demo:
        name = gr.Textbox(label="Name")
        output = gr.Textbox(label="Output Box")
        greet_btn = gr.Button("Greet")
        greet_btn.click(fn=greet, inputs=name, outputs=output, api_name="greet")

    demo.launch()
#########################################################################################


def example2():
    def welcome(name):
        return f"Welcome to Gradio, {name}!"

    with gr.Blocks() as demo:
        gr.Markdown(
        """
        # Hello World!
        Start typing below to see the output.
        """)
        inp = gr.Textbox(placeholder="What is your name?")
        out = gr.Textbox()
        inp.change(welcome, inp, out)

    demo.launch()
##############################################################################

def example3():
    def increase(num):
        return num + 1

    with gr.Blocks() as demo:
        a = gr.Number(label="a")
        b = gr.Number(label="b")
        atob = gr.Button("a > b")
        btoa = gr.Button("b > a")
        atob.click(increase, a, b)
        btoa.click(increase, b, a)

    demo.launch()


def example4():
    def slow_echo(message, history):
        for i in range(len(message)):
            time.sleep(0.3)
            yield "You typed: " + message[: i + 1]

    gr.ChatInterface(slow_echo).queue().launch()


def example5():
    def yes_man(message, history):
        if message.endswith("?"):
            return "Yes"
        else:
            return "Ask me anything!"

    gr.ChatInterface(
        yes_man,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
        title="Yes Man",
        description="Ask Yes Man any question",
        theme="soft",
        examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
        cache_examples=True,
        retry_btn=None,
        undo_btn="Delete Previous",
        clear_btn="Clear",
    ).launch()


def example6():
    def echo(message, history, system_prompt, tokens):
        response = f"System prompt: {system_prompt}\n Message: {message}."
        for i in range(min(len(response), int(tokens))):
            time.sleep(0.05)
            yield response[: i + 1]

    demo = gr.ChatInterface(echo,
                            additional_inputs=[
                                gr.Textbox("You are helpful AI.", label="System Prompt"),
                                gr.Slider(10, 100)
                            ]
                            )

    demo.queue().launch()


def example7():
    def echo(message, history, system_prompt, tokens):
        response = f"System prompt: {system_prompt}\n Message: {message}."
        for i in range(min(len(response), int(tokens))):
            time.sleep(0.05)
            yield response[: i + 1]

    with gr.Blocks() as demo:
        system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
        slider = gr.Slider(10, 100, render=False)

        gr.ChatInterface(
            echo, additional_inputs=[system_prompt, slider]
        )

    demo.queue().launch()

if __name__ == '__main__':
    example7()