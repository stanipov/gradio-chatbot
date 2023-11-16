import logging
import torch
from transformers import BitsAndBytesConfig
##################################  Code  ##################################

def isClose(a,b, eps=1e-4):
    """ Compares 2 float numbers if they are close """
    return abs(a-b) <=eps

def merge2ChatML(message, history, sys_prompt):
    """ Merges user message, history, and system prompt into ChatML format """
    messages = [{'role': 'system', 'content': sys_prompt}]
    for usr, ai in history:
        messages.append({'role': 'user', 'content': usr})
        messages.append({'role': 'assistant', 'content': ai})

    messages.append({'role': 'assistant', 'content': message})
    return messages


def verify_config(cfg):
    """ A simple check of the model config """
    logger = logging.getLogger(__name__)
    if 'base_model' not in cfg:
        logger.critical('No base model provided!')
        return False
    elif cfg['base_model'] is None or cfg['base_model'] == '':
        logger.critical('Base model is empty!')
        return False
    return True

##################################  Parse config for LLM  ##################################
def get_4bit_bnb_config(cfg):
    """ Builds simple 4 bits BnB quantization config """
    ans = {}
    ans['bnb_4bit_quant_type'] = cfg['quantization'].get('bnb_4bit_quant_type', 'nf4')
    ans['bnb_4bit_use_double_quant'] = cfg['quantization'].get('bnb_4bit_use_double_quant', True)
    dtype = cfg['quantization'].get('bnb_4bit_compute_dtype', 'bf16').lower()
    if dtype=='bf16':
        dtype = torch.bfloat16
    elif dtype=='fp16':
        dtype = torch.float16
    else:
        dtype = None
    ans['dtype'] = dtype

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=ans['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype= ans['dtype'],
        bnb_4bit_use_double_quant=ans['bnb_4bit_use_double_quant']
    )

    return bnb_config

def parse_config(cfg):
    """ Parses the model config """
    ans = {}
    logger = logging.getLogger(__name__)
    ans['base_model'] = cfg['base_model']

    _t = cfg.get('adapter', None)
    ans['adapter'] = _t if _t != '' else None
    msg = f'Using adapter.' if ans['adapter'] else 'Adapter is not provided.'
    logger.info(msg)

    _t = cfg.get('cache_dir', None)
    ans['cache_dir'] = _t if _t != '' else None
    msg = f'Using cache dir: {ans["cache_dir"]}' if ans['cache_dir'] else 'Using default cache dir.'
    logger.info(msg)

    if 'quantization' in cfg:
        logger.info('Building simple 4 bits BnB config')
        ans['quant'] = get_4bit_bnb_config(cfg)
    else:
        logger.info('No quantization for loading')
        ans['quant'] = None

    ans['dev_map'] = cfg.get('device_map', 'auto')
    _t = cfg.get('torch_dtype', None)
    if _t:
        if _t.lower() == 'fp16':
            _t = torch.float16
        elif _t.lower() == 'bf16':
            _t = torch.bfloat16
        else:
            _t = 'auto'
    ans['load_dtype'] = _t
    return ans