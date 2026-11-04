from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def make_prompt_encoder(model_path, model_config):
    '''
    Returns a function that turns a prompt string into a vLLM prompt. By default the string is passed as is. With
    "tokenize_prompts": true in the model config, it is tokenized with the model's tokenizer.json, which parses
    control tokens such as [INST] in the text, and passed as token ids. Needed for models loaded with vLLM's
    Mistral tokenizer (tokenizer_mode "mistral"), which encodes control tokens in a string prompt as plain text.
    The prompt templates in the config must then include the BOS token themselves.
    '''
    if not model_config.get("tokenize_prompts"):
        return lambda text: text
    tokenizer = Tokenizer.from_file(hf_hub_download(model_path, "tokenizer.json"))
    return lambda text: {"prompt_token_ids": tokenizer.encode(text, add_special_tokens=False).ids}
