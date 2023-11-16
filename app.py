#!/ext4/pyenv/llm/bin/python

import gradio as gr
import time
import logging
import datetime as dt
import sys
from functools import partial

from src.model import Mistral_chat_LLM
from src.utils import merge2ChatML
from threading import Thread

def echo(message, history, system_prompt, tokens, T):
    response = f"Temperature: {T}\nSystem prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]

def __generate_core(message, history, llm, system_prompt, output_type):
    # Generate prompt
    logger = logging.getLogger(__name__)
    #logger.debug(f'Message: {message}')
    #logger.debug(f'History: {history}')
    #logger.debug(f'System prompt: {system_prompt}')
    messages = merge2ChatML(message, history, system_prompt, )
    prompt = llm.conv_messages_prompt(messages)
    #logger.debug(f'Prompt: {prompt}')
    response = llm.generate(prompt=prompt, kind=output_type)
    return response


def generate_core(message, history, system_prompt, llm):
    # Generate prompt
    messages = merge2ChatML(message, history, system_prompt)
    prompt = llm.conv_messages_prompt(messages)

    llm.logger.debug(f'PROMPT: {prompt}')
    tokenized_prompt = llm.tokenizer([prompt], return_tensors="pt").to(llm.device)
    llm.logger.debug(f'TOKENIZED PROMPT: {tokenized_prompt}')
    streamer_kw = dict(
        tokenized_prompt,
        streamer=llm.streamer,
        max_new_tokens=llm.max_new_tokens,
        do_sample=llm.do_sample,
        top_p=llm.top_k,
        top_k=llm.top_k,
        temperature=llm.T,
        num_beams=llm.num_beams
    )
    t = Thread(target=llm.model.generate, kwargs=streamer_kw)
    t.start()

    partial_message = ""
    for new_token in llm.streamer:
        if new_token != llm.bos_tok or new_token != llm.eos_tok:
            partial_message += new_token
            yield partial_message



def start_server(llm, server_port: int = 5050):
    logger = logging.getLogger(__name__)

    with gr.Blocks() as demo:
        _system_prompt = gr.Textbox("You are helpful AI chatbot.", label="System Prompt", interactive=True)
        system_prompt = _system_prompt.value
        with gr.Accordion(label='Parameters', open=False) as acc1:
            with gr.Row() as row1:
                top_p = float(gr.Number(label="Top p", value=0.95, interactive=True, render=True).value)
                top_k = int(gr.Number(label="Top k", value=250, interactive=True, render=True).value)
                T = float(gr.Number(label="Temperature", value=0.5, interactive=True, render=True).value)
                num_beams = int(gr.Number(label="Beams", value=1, interactive=True, render=True).value)
                penalty_alpha = float(gr.Number(label="Repetition penalty", value=1.5, interactive=True, render=True).value)
                max_new_tokens = int(gr.Number(label="# new tokens", value=512, interactive=True, render=True).value)
                do_sample = bool(gr.Checkbox(value=True, interactive=True, label='Do sampling', render=True).value)
                base_llm = bool(gr.Checkbox(value=False, interactive=True, label='Base LLM', render=True).value)
                output_type = str(gr.Dropdown(interactive=True, label='Output behavior', value='streaming',
                                               render=True, choices=['streaming', 'pipe']).value)
        logger.debug(f'Output behavior: {output_type}')
        logger.debug(f'max_new_tokens: {max_new_tokens}')

        # unset/set adapter
        if base_llm:
            llm.disable_adapter()
        else:
            llm.enable_adapter()

        # set up generation part:
        if output_type=='streaming':
            llm.set_streamer(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)
        elif output_type=='pipe':
            llm.set_gen_pipeline(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)
        else:
            logger.warning(f'Output behavior "{output_type}" is not recognized. Rolling back to streaming.')
            output_type == 'streaming'
            llm.set_streamer(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)

        #generate = partial(generate_core,
        #                   llm=llm,
        #                   system_prompt=system_prompt,
        #                   output_type=output_type
        #                   )


        gr.ChatInterface(
            generate_core,
            additional_inputs=[llm, system_prompt]
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
    llm.from_pretrained(enable_adapter=True)

    #start_server(llm, serv_port, out_behavior)
    with gr.Blocks() as demo:
        _system_prompt = gr.Textbox("You are helpful AI chatbot.", label="System Prompt", interactive=True)
        system_prompt = _system_prompt.value
        with gr.Accordion(label='Parameters', open=False) as acc1:
            with gr.Row() as row1:
                top_p = float(gr.Number(label="Top p", value=0.95, interactive=True, render=True).value)
                top_k = int(gr.Number(label="Top k", value=250, interactive=True, render=True).value)
                T = float(gr.Number(label="Temperature", value=0.5, interactive=True, render=True).value)
                num_beams = int(gr.Number(label="Beams", value=1, interactive=True, render=True).value)
                penalty_alpha = float(
                    gr.Number(label="Repetition penalty", value=1.5, interactive=True, render=True).value)
                max_new_tokens = int(gr.Number(label="# new tokens", value=512, interactive=True, render=True).value)
                do_sample = bool(gr.Checkbox(value=True, interactive=True, label='Do sampling', render=True).value)
                base_llm = bool(gr.Checkbox(value=False, interactive=True, label='Base LLM', render=True).value)
                output_type = str(gr.Dropdown(interactive=True, label='Output behavior', value='streaming',
                                              render=True, choices=['streaming', 'pipe']).value)
        logger.debug(f'Output behavior: {output_type}')
        logger.debug(f'max_new_tokens: {max_new_tokens}')

        # unset/set adapter
        if base_llm:
            llm.disable_adapter()
        else:
            llm.enable_adapter()

        # set up generation part:
        if output_type == 'streaming':
            llm.set_streamer(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)
        elif output_type == 'pipe':
            llm.set_gen_pipeline(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)
        else:
            logger.warning(f'Output behavior "{output_type}" is not recognized. Rolling back to streaming.')
            output_type == 'streaming'
            llm.set_streamer(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha)

        generate = partial(generate_core, llm=llm)

        gr.ChatInterface(
            generate,
            additional_inputs=[_system_prompt]
        )

    demo.queue().launch(server_port=serv_port)



