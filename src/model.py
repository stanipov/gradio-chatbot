from src.utils import isClose, verify_config, parse_config, get_4bit_bnb_config

import sys
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextIteratorStreamer
)
from peft import PeftConfig
from threading import Thread

##################################  Code  ##################################

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
        tok_name = ''
        if self.cfg['tokenizer']:
            tok_name = self.cfg['tokenizer']
        else:
            tok_name = self.cfg['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=tok_name,
                                                       cache_dir=self.cfg['cache_dir'], padding_size="left")
        self.model.resize_token_embeddings(len(self.tokenizer))
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
            self.logger.info('Adapter enabled.')

    def disable_adapter(self):
        """ Disables adapter """
        if self.cfg['adapter']:
            self.model.disable_adapters()
            self.logger.info('Adapter disabled.')
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
                max_new_tokens=max_new_tokens,
                repetition_penalty=penalty_alpha
            )
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens
            self.penalty_alpha = penalty_alpha
            self.logger.info('Updated generation pipeline.')

    def set_streamer(self, T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha, timeout=20):
        """ Sets a streamer for generation """
        if self._are_new_args(T, top_k, top_p, do_sample, num_beams, max_new_tokens, penalty_alpha):
            self.streamer = TextIteratorStreamer(self.tokenizer,
                                                  timeout=timeout,
                                                  skip_prompt=True,
                                                  skip_special_tokens=True)
            self.T = T
            self.top_k = top_k
            self.top_p = top_p
            self.do_sample = do_sample
            self.num_beams = num_beams
            self.max_new_tokens = max_new_tokens
            self.penalty_alpha = penalty_alpha
            self.logger.info('Updated streamer.')

    def gen_stream(self, prompt: str):
        """ Generation streamer """
        self.logger.debug(f'PROMPT: {prompt}')
        tokenized_prompt = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        self.logger.debug(f'TOKENIZED PROMPT: {tokenized_prompt}')
        streamer_kw = dict(
            tokenized_prompt,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_k,
            top_k=self.top_k,
            temperature=self.T,
            num_beams=self.num_beams,
            repetition_penalty=self.penalty_alpha
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
