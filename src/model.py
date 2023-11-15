"""
base_model.enable_adapters()
max_length = 2048
#gen_pipe = pipeline('text-generation', model=base_model, tokenizer=tokenizer, temperature = 0.85, do_sample=True, max_new_tokens = 256)
do_sample=True
top_p=0.95
top_k=100
temperature=0.5
num_beams=1
penalty_alpha=0.6
max_new_tokens=1024
gen_pipe = pipeline('text-generation', model=base_model, tokenizer=tokenizer,
                    penalty_alpha=penalty_alpha,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    max_new_tokens=1024)





"""

from src.utils import isClose

import sys
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    TextIteratorStreamer
)
from peft import PeftConfig
from threading import Thread

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
########################################################################################################################

class __old_Mistral_chat_LLM:
    def __init__(self, config):
        """ Parses the config """
        self.logger = logging.getLogger(__name__)
        if not verify_config(config):
            self.logger.critical('Config is not correct!')
            sys.exit(-1)
        else:
            self.logger.info(f'Config is OK')
        self.cfg = parse_config(config)
        self.top_k = -1
        self.top_p = -1
        self.T = -1
        self.max_new_tokens = -1
        self.num_beams = -1
        self.do_sample = False
        self.penalty_alpha = -1
        self.adapter_change = False
        self.pipe = None
        self.streamer = None
        self.device = 'cpu'
        self.bos_tok = "<|im_start|>"
        self.eos_tok = "<|im_end|>\n"

    def load_base_model(self):
        """ Loads the base model """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg['base_model'],
            cache_dir=self.cfg['cache_dir'],
            quantization_config=self.cfg['quant'],
            device_map=self.cfg['dev_map'],
            torch_dtype=self.cfg['load_dtype']
        )
        self.device = self.model.device

    def load_tokenizer(self):
        """ Loads tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.cfg['base_model'],
                                          cache_dir=self.cfg['cache_dir'],  padding_size="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_adapter(self):
        """ Adds adapter """
        if self.cfg['adapter']:
            peft_config2 = PeftConfig.from_pretrained(self.cfg['adapter'])
            self.model.add_adapter(peft_config2)

    def enable_adapter(self):
        """ Enables adapter """
        if self.cfg['adapter']:
            self.model.enable_adapters()
            self.adapter_change = True

    def disable_adapter(self):
        """ Disables adapter """
        if self.cfg['adapter']:
            self.model.enable_adapters()
            self.adapter_change = True

    def _is_new_pipeline(self, T: float,  top_k: int, top_p: float,
                         do_sample: bool, num_beams: int,
                         max_new_tokens: int,
                         penalty_alpha: float):
        """ Checks if new pipeline should be created """
        if self.adapter_change:
            self.adapter_change = False
            return True
        if not isClose(self.T, T):
            return True
        if self.top_k != top_k:
            return True
        if not isClose(self.top_p, top_p):
            return True
        if do_sample != self.do_sample:
            return True
        if max_new_tokens != self.max_new_tokens:
            return True
        if not isClose(self.penalty_alpha, penalty_alpha):
            return True
        return False

    def set_gen_pipeline(self, T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
        """
        Sets if necessary a new generation pipeline
        """
        if self._is_new_pipeline(top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
            self.pipe = pipeline(
                task='text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=T,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens
            )
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens


    def set_streamer(self, T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha, timeout=20):
        """ Sets a streamer for generation """
        if self._is_new_pipeline(top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
            self.streamer =  TextIteratorStreamer(self.tokenizer,
                                                  timeout=timeout,
                                                  skip_prompt=True,
                                                  skip_special_tokens=True)
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens


    def gen_stream(self, prompt: str):
        """ Generation streamer """
        tokenized_prompt = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        streamer_kw = dict(
            tokenized_prompt,
            streamer=self.streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=self.top_k,
            top_k=self.top_k,
            temperature=1.0,
            num_beams=1
        )
        t = Thread(target=self.model.generate, kwargs=streamer_kw)
        t.start()

        partial_message = ""
        for new_token in self.streamer:
            if new_token != self.bos_tok or new_token != self.eos_tok:
                partial_message += new_token
                yield partial_message

    def generate(self, prompt: str, kind='streaming'):
        if kind == 'streaming':
            return self.gen_stream(prompt)
        elif kind == 'pipe':
            result = self.pipe(prompt)
            return result[0]['generated_text'].replace(prompt, '')

    def __old_messages2prompt(self, messages):
        """ Converts ChatML messages to a prompt """
        prompt = ''
        for line in messages:
            prompt += f'{self.bos_tok}{line["role"]}\n{line["content"]}{self.eos_tok}'
        return prompt

    def messages2prompt(self, messages):
        """ Converts ChatML messages to a prompt """
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def prompt4gen(self, prompt):
        """ Adds extra for open-ended generation """
        prompt += f'{self.bos_tok}assistant\n'
        return prompt
########################################################################################################################

class BaseLLM_SingleAdapter:
    """
    Base class for a LLM model with a single adapter
    """
    def __init__(self, config):
        """ Parses the config """
        self.logger = logging.getLogger(__name__)
        if not verify_config(config):
            self.logger.critical('Config is not correct!')
            sys.exit(-1)
        else:
            self.logger.info(f'Config is OK')
        self.cfg = parse_config(config)
        self.top_k = -1
        self.top_p = -1
        self.T = -1
        self.max_new_tokens = -1
        self.num_beams = -1
        self.do_sample = False
        self.penalty_alpha = -1
        self.adapter_change = False
        self.pipe = None
        self.streamer = None
        self.device = 'cpu'
        self.bos_tok = ""
        self.eos_tok = ""

    def from_pretrained(self, enable_adapter=True):
        self.logger.info('Loading base model')
        self.__load_base_model()
        self.logger.info('Loading tokenizer')
        self.__load_tokenizer()
        if enable_adapter:
            self.add_adapter()
            self.enable_adapter()

    def __load_base_model(self):
        """ Loads the base model """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg['base_model'],
            cache_dir=self.cfg['cache_dir'],
            quantization_config=self.cfg['quant'],
            device_map=self.cfg['dev_map'],
            torch_dtype=self.cfg['load_dtype']
        )
        self.device = self.model.device

    def __load_tokenizer(self):
        """ Loads tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.cfg['base_model'],
                                                       cache_dir=self.cfg['cache_dir'], padding_size="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_adapter(self):
        """ Adds adapter """
        if self.cfg['adapter']:
            peft_config2 = PeftConfig.from_pretrained(self.cfg['adapter'])
            self.model.add_adapter(peft_config2)
            self.logger.info('Adapter added')

    def enable_adapter(self):
        """ Enables adapter """
        if self.cfg['adapter']:
            self.model.enable_adapters()
            self.adapter_change = True
            self.logger.info('Adapter enabled')

    def disable_adapter(self):
        """ Disables adapter """
        if self.cfg['adapter']:
            self.model.enable_adapters()
            self.adapter_change = True

    def _are_new_args(self, T: float,  top_k: int, top_p: float,
                         do_sample: bool, num_beams: int,
                         max_new_tokens: int,
                         penalty_alpha: float):
        """ Checks if new pipeline should be created """
        if self.adapter_change:
            self.adapter_change = False
            return True
        if not isClose(self.T, T):
            return True
        if self.top_k != top_k:
            return True
        if not isClose(self.top_p, top_p):
            return True
        if do_sample != self.do_sample:
            return True
        if num_beams != self.num_beams:
            return True
        if max_new_tokens != self.max_new_tokens:
            return True
        if not isClose(self.penalty_alpha, penalty_alpha):
            return True
        return False

    def set_gen_pipeline(self, T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
        """
        Sets if necessary a new generation pipeline
        """
        if self._are_new_args(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
            self.pipe = pipeline(
                task='text-generation',
                model=self.model,
                tokenizer=self.tokenizer,
                temperature=T,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens
            )
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens
            self.logger.info('Updated generation pipeline.')

    def set_streamer(self, T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha, timeout=20):
        """ Sets a streamer for generation """
        if self._are_new_args(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
            self.streamer =  TextIteratorStreamer(self.tokenizer,
                                                  timeout=timeout,
                                                  skip_prompt=True,
                                                  skip_special_tokens=True)
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens
            self.logger.info('Updated streamer.')

    def gen_stream(self, prompt: str):
        """ Generation streamer """
        tokenized_prompt = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        streamer_kw = dict(
            tokenized_prompt,
            streamer=self.streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=self.top_k,
            top_k=self.top_k,
            temperature=1.0,
            num_beams=1
        )
        t = Thread(target=self.model.generate, kwargs=streamer_kw)
        t.start()

        partial_message = ""
        for new_token in self.streamer:
            if new_token != self.bos_tok or new_token != self.eos_tok:
                partial_message += new_token
                yield partial_message

    def generate(self, prompt: str, kind='streaming'):
        if kind == 'streaming':
            return self.gen_stream(prompt)
        elif kind == 'pipe':
            result = self.pipe(prompt)
            return result[0]['generated_text'].replace(prompt, '')

    def conv_messages_prompt(self, messages):
        """ Converts ChatML messages to prompt for generation """
        raise NotImplementedError('Implement this this method for your specific model!')


########################################################################################################################
class Mistral_chat_LLM(BaseLLM_SingleAdapter):
    def __init__(self, cfg):
        super().__init__(cfg)
        # specifics for generating prompt for chat
        # Tokens that do indicate bot response
        self.bos_tok = "<|im_start|>"
        self.eos_tok = "<|im_end|>\n"

    def conv_messages_prompt(self, messages):
        """ Converts ChatML messages to prompt for generation """
        core_prompt = self.__messages2prompt(messages)
        return self.__prompt4gen(core_prompt)

    def __messages2prompt(self, messages):
        """ Converts ChatML messages to a prompt """
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def __prompt4gen(self, prompt):
        """ Adds extra for open-ended generation """
        prompt += f'{self.bos_tok}assistant\n'
        return prompt
