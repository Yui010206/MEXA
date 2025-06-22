import requests
import openai
from openai import OpenAI
from transformers import AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration, AutoModelForCausalLM, Gemma3ForConditionalGeneration
import torch
import transformers
import pdb
from pprint import pprint
import time


def get_model(args):
    if 'dummy' in args.model:
        model = DummyModel()
        
    elif len(args.endpoint) > 0:
        if 'openai' in args.endpoint.lower():
            model = GPT(args.api_key, args.endpoint, args.model)
                    
        elif 'deepseek' in args.endpoint.lower() or 'lambda' in args.endpoint.lower():
            model = DeepSeek(args.api_key, args.api_url, args.model)
            
        elif 'gemini' in args.model or 'gemma' in args.model:
            model = Gemini(args.api_key, args.model)
        else:
            raise NotImplementedError(f"Model {args.model} with endpoint {args.endpoint} not implemented")
    elif 'gpt' in args.model or 'deepseek' in args.model:
        model = GPT(args.api_key, args.api_url, args.model)
    elif 'Llama-2' in args.model:
        model = LLaMA2(args.model, 1.0)
    elif 'Llama-3' in args.model:
        model = LLaMA3(args.model, 1.0) 
    elif 'Llama-4' in args.model:
        model = LLaMA4(args.model) 
    elif 'gemma' in args.model.lower():
        model = Gemma(args.model)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    return model


class DummyModel():
    def __init__(self):
        super().__init__()
    def forward(self, head, prompts):
        output = {
            'response': "dummy response",
            'reasoning_content': "dummy reasoning",
            'usage': "dummy usage",
            'message': "dummy message",
        }
        return output 
    

class DeepSeek():
    def __init__(self, api_key, api_url, model_name):
        super().__init__()
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def forward(self, head, prompts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        messages = []
        for i, prompt in enumerate(prompts):
            messages.append(
                {"role": "user", "content": prompt}
            )
            data = {
                "model": self.model_name,
                "messages": messages
            }
            try:
                response = requests.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()  # Raise error for bad responses
                response_json = response.json()
            except Exception as e:
                print(f"Error in model. Response: {response}")
                time.sleep(5)
                return 
            response_text = response_json["choices"][0]["message"]["content"]
            reasoning_content = response_json["choices"][0]["message"]["reasoning_content"] if "reasoning_content" in response_json["choices"][0]["message"] else ""
            messages.append(
                {"role": "assistant", "content": response_text}
            )
            usage = dict(response_json['usage'])  # completion_tokens, prompt_tokens, total_tokens
        output = {
            'response': messages[-1]["content"],
            'reasoning_content': reasoning_content,
            'usage': usage,
            'message': messages,
        }
        return output


import os
from openai import AzureOpenAI
from dotenv import load_dotenv # pip install python-dotenv

load_dotenv()

class GPT():
    def __init__(self, api_key, endpoint, model_name):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.endpoint = endpoint 

    def forward(self, head, prompts):
        
        client = AzureOpenAI(
            azure_endpoint = self.endpoint, 
            api_key=self.api_key,  
            api_version="2024-02-01"
        )
        
        # print(self.api_key)
        # print(self.endpoint)
        # print(self.model_name)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages = [
                {"role": "system", "content": "You are a helpful expert in video analysis."},
                {"role": "user", "content": prompts[0]},
            ]
        )
        
        output = {
            'response': response.choices[0].message.content,
            'message': prompts,
        }
        
        return output    

class Gemini():
    def __init__(self, api_key, model_name):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name

    def forward(self, head, prompts):
        from google import genai
        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompts[0],
        )

        output = {
            'response': response.text,
            'message': prompts,
        }
        return output


class LLaMA2():
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer,
            temperature=temperature
        )

    def forward(self, head, prompts):
        prompt = prompts[0]
        sequences = self.pipeline(
            prompt,
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        response = sequences[0]['generated_text']  # str
        info = {
            'message': prompt,
            'response': response
        }
        return info['response'], info


class LLaMA3():
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=model_name, 
            model_kwargs={"torch_dtype": torch.float16}, 
            device_map="auto",
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def forward(self, head, prompts):
        prompt = prompts[0]
        messages = [
            {"role": "system", "content": head},
            {"role": "user", "content": prompt}
        ]
        sequences = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
            temperature=self.temperature,
        )
        response = sequences[0]["generated_text"][-1]["content"]
        info = {
            'message': prompt,
            'response': response
        }
        return info['response'], info
    
class LLaMA4():
    def __init__(self, model_name, max_new_tokens=256):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # cache_dir = cache_dir,
            # use_auth_token = use_auth_token
        )
        self.max_new_tokens = max_new_tokens

    def forward(self, head, prompts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]},
                ]
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            attn_implementation="eager",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )

        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        output = {
            'response': response,
            'message': messages,
        }
        return output


class Gemma():
    def __init__(self, model_name, max_new_tokens=128):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name, device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def forward(self, head, prompts):
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
        #     },
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": prompts[0]}
        #         ]
        #     }
        # ]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts[0]}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        output = {
            'response': decoded,
            'message': messages,
        }
        return output