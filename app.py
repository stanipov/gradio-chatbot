#!/ext4/pyenv/llm/bin/python

import gradio as gr
import time
import logging
import datetime as dt
import sys
from functools import partial

from src.model import Mistral_chat_LLM
from src.utils import merge2ChatML

def echo(message, history, system_prompt, tokens, T):
    response = f"Temperature: {T}\nSystem prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]

def generate_core(llm, message, history, system_prompt,
                  top_p, top_k, T, penalty_alpha,
                  max_new_tokens, do_sample,
                  output_type):
    # form ChatML
    messages = merge2ChatML(message, history, system_prompt)



    pass


def start_server(llm, server_port: int=5050, output_type:str='streaming'):

    with gr.Blocks() as demo:
        system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")

        with gr.Row() as row1:
            top_p = gr.Number(label="Top p", value=0.95, interactive=True, render=True)
            top_k = gr.Number(label="Top k", value=250, interactive=True, render=True)
            T = gr.Number(label="Temperature", value=0.5, interactive=True, render=False)
            penalty_alpha = gr.Number(label="Repetition penalty", value=1.0, interactive=True, render=True)
            max_new_tokens = gr.Number(label="# new tokens", value=1024, interactive=True, render=True)
            do_sample = gr.Checkbox(value=True, interactive=True, label='Do sampling', render=True)

        generate = partial(generate_core,
                           llm=llm,
                           system_prompt=system_prompt,
                           top_p=top_p,
                           top_k=top_k,
                           T=T,
                           penalty_alpha=penalty_alpha,
                           max_new_tokens=max_new_tokens,
                           do_sample=do_sample,
                           output_type=output_type
                           )
        gr.ChatInterface(
            echo
        )

    demo.queue().launch(server_port=server_port)

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
    # https://www.gradio.app/guides/creating-a-chatbot-fast
    # https://www.gradio.app/guides/four-kinds-of-interfaces
    # https://www.gradio.app/docs/interface


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

    serv_port = config['app'].get('port', 5050)
    out_behavior = config['app'].get('output_behavior', 'streaming')

    llm = Mistral_chat_LLM(config['model'])
    llm.load_base_model()
    start_server(llm, serv_port, out_behavior)




