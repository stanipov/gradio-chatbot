#!/ext4/pyenv/llm/bin/python

import gradio as gr
import time
import logging
import datetime as dt
import sys

from src.models.load import LLM

def echo(message, history, system_prompt, tokens, T):
    response = f"Temperature: {T}\nSystem prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]


def start_server(*args):
    SERVER_PORT = 5050

    def my_function(x, progress=gr.Progress()):
        progress(0, desc="Starting...")
        time.sleep(1)
        for i in progress.tqdm(range(100)):
            time.sleep(0.1)
        return x

    with gr.Blocks() as demo:
        system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
        T = gr.Number(label="Temperature", value=0.5, interactive=True, render=False)
        with gr.Row() as row1:
            top_p = gr.Number(label="Top p", value=0.85, interactive=True, render=True)
            do_sample = gr.Checkbox(value=True, interactive=True, label='Do sampling', render=True)

        #with gr.Row():
        #    T = gr.Number(label="Temperature", value=0.5, interactive=True, render=False)
        #    do_sample = gr.Checkbox(value=True, interactive=True, label='Do sampling', render=False)

        slider = gr.Slider(25, 250, render=False)

        gr.Interface(my_function, gr.Textbox(), gr.Textbox())

        gr.ChatInterface(
            echo,
            additional_inputs=[system_prompt, slider, T]
            #additional_inputs=[system_prompt, slider]
        )

    demo.queue().launch(server_port=SERVER_PORT)

def set_logger():
    """ Sets up a stdout logger """
    today = dt.datetime.today()
    dt_str = f"{today.month:02d}-{today.day:02d}-{today.year}"

    logFormatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)8s] [%(levelname)-8s] %(message)s"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logFormatter)
    ch.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(ch)
    logger = logging.getLogger(__name__)
    return logger


##########################################################################################################
if __name__ == '__main__':
    import json
    import argparse

    arg_desc = '''\
        Config file location. Required parameter!
        '''

    logger = set_logger()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=arg_desc)

    parser.add_argument("-cfg", metavar="config file", help="Path to the config", required=True)
    args = parser.parse_args()
    cfg_file = args.cfg

    msg = """
    ===================================================================
        Starting a simple local LLM chatbot wrapper for experiments.
    =================================================================== 
    """
    logger.info(msg)
    with open(cfg_file, 'r') as f:
        config = json.load(f)

    if 'model' not in config:
        logger.critical(f'Config file does not contain model config!')
        sys.exit(1)

    llm = LLM(config['model'])
    llm.load_base_model()
    #start_server()




