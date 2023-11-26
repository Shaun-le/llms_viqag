import torch
from peft import PeftModel
import transformers
import textwrap
import fire
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, DataCollatorForSeq2Seq
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
import os
import json
import os.path as osp
from typing import Union
from datasets import load_dataset
from tqdm import tqdm


def test(
        base_model: str = 'manhtt-079/llama-2-7b',
        lora_weight: str = '/home/int2-user/llms_viqag/lora/weight-vinewsqa',
        check_point: str = '',
        test_path: str = '',
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, lora_weight, torch_dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model = model.eval()
    model = torch.compile(model)

    def generate_prompt(data_point):
        return f"""Dưới đây là một Instruction mô tả một nhiệm vụ, kèm theo một Input cung cấp ngữ cảnh. Viết một Response hoàn thành yêu cầu một cách thích hợp.
        
        ### Instruction:
        {data_point["instruction"]}
        
        ### Input:
        {data_point["input"]}
        
        ### Response:
        """

    CUTOFF_LEN = 1024

    '''def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < CUTOFF_LEN
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result'''

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenizer(full_prompt)
        return tokenized_full_prompt

    test = load_dataset("json", data_files=test_path)
    test_data = (test["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=['instruction','input','output']))

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, model=model, return_tensors="pt", padding=True
    )
    dataloader = torch.utils.data.DataLoader(test_data, collate_fn=data_collator, batch_size=128)

    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
    )

    dec = []
    predictions = []

    for _, batch in enumerate(tqdm(dataloader)):
        outputs = model.generate(
            input_ids=batch['input_ids'].to('cuda'),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )

        with tokenizer.as_target_tokenizer():
            outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in
                       outputs.sequences]

        dec.append(outputs)

    for i in range(len(dec)):
        predictions.extend(dec[i][0].split("### Response:")[1].strip())

    with open(os.path.join(check_point, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'predictions': predictions}, f, ensure_ascii=False, indent=2)


    print(predictions[0])

if __name__ == "__main__":
    fire.Fire(test)