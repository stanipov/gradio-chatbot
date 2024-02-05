import gradio as gr
#import time
import logging
import datetime as dt
import sys
from functools import partial

from src.model import Mistral_chat_LLM
from src.utils import merge2ChatML
from threading import Thread

def generate_core(message, history,
                  system_prompt, top_p,
                  top_k, T, num_beams, penalty_alpha,
                  max_new_tokens, do_sample,
                  base_llm, llm):

    """ Streaming generation """
    # Generate prompt
    messages = merge2ChatML(message, history, system_prompt)
    prompt = llm.conv_messages_prompt(messages)

    # unset/set adapter
    if base_llm:
        llm.disable_adapter()
    else:
        llm.enable_adapter()

    top_k = int(top_k)
    top_p = float(top_p)
    num_beams = int(num_beams)
    max_new_tokens = int(max_new_tokens)
    penalty_alpha = float(penalty_alpha)

    llm.set_streamer(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)
    tokenized_prompt = llm.tokenizer([prompt], return_tensors="pt").to(llm.device)
    streamer_kw = dict(
        tokenized_prompt,
        streamer=llm.streamer,
        max_new_tokens=llm.max_new_tokens,
        do_sample=llm.do_sample,
        top_p=llm.top_k,
        top_k=llm.top_k,
        temperature=llm.T,
        num_beams=llm.num_beams,
        repetition_penalty=penalty_alpha
    )
    t = Thread(target=llm.model.generate, kwargs=streamer_kw)
    t.start()

    partial_message = ""
    for new_token in llm.streamer:
        if new_token != llm.bos_tok or new_token != llm.eos_tok:
            partial_message += new_token
            yield partial_message


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

    serv_port = config['app'].get('port', 5050)
    out_behavior = config['app'].get('output_behavior', 'streaming')

    llm = Mistral_chat_LLM(config['model'])
    llm.from_pretrained(enable_adapter=True)

    default_sys_prompt = config['generation'].get('system_prompt', "You are sarcastic chatbot. You respond to a prompt from a user.")

    with gr.Blocks(theme=gr.themes.Soft()) as ChatBotInterface:
        system_prompt = gr.Textbox(default_sys_prompt, label="System Prompt", interactive=True)
        with gr.Accordion(label='Parameters', open=False) as acc1:
            with gr.Row() as row1:
                top_p = gr.Number(label="Top p", value=0.95, interactive=True, render=True)
                top_k = gr.Number(label="Top k", value=250, interactive=True, render=True)
                T = gr.Number(label="Temperature", value=0.5, interactive=True, render=True)
                num_beams = gr.Number(label="Beams", value=1, interactive=True, render=True)
                penalty_alpha = gr.Number(label="Repetition penalty", value=1.5, interactive=True, render=True)
                max_new_tokens = gr.Number(label="# new tokens", value=512, interactive=True, render=True)
                do_sample = gr.Checkbox(value=True, interactive=True, label='Do sampling', render=True)
                base_llm = gr.Checkbox(value=False, interactive=True, label='Base LLM', render=True)

        generate = partial(generate_core, llm=llm)

        gr.ChatInterface(
            generate,
            additional_inputs=[system_prompt, top_p,
                 top_k, T, num_beams, penalty_alpha,
                 max_new_tokens, do_sample,
                 base_llm],

        )

    ChatBotInterface.queue().launch(server_port=serv_port, server_name='0.0.0.0', show_api=False)



