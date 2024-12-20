import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline
from accelerate import Accelerator

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="myselfrew/llama3_math_test_tmp07",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="llama3_math_test_tmp07_orm_1e6.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="1231czx/llama3_orm_1e6",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=4,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 1,
}
reward_model = script_args.reward_name_or_path
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model,
    device=device,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    truncation=True,
)

output_name = "1231czx/" + script_args.output_dir.replace(".json", "")
ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
if 'DS' in ds_dir or 'Deep' in ds_dir:
    ds = load_dataset(ds_dir, split='test')
else:
    ds = load_dataset(ds_dir, split="train")

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

"""
We process the data format here and query the reward model to get the rewards.
"""


def get_reward(test_texts, all_len):
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = []
    i = 0
    for output in pipe_outputs:
        rewards.append(output[0]['score']) #- all_len[i] * 0.001)
        i += 1
    #rewards = [output[0]["score"] for output in pipe_outputs]
    return rewards


def change_of_format(prom, resp):
    # To be modified according to the reward model and the LLM you use
    # Be careful about multi-turn conversions
    """
    prom = prom.replace("<s>GPT4 Correct User: ", "").replace("<|end_of_turn|>GPT4 Correct Assistant:", "")

    final_resp = resp.split("GPT4 Correct User")[0]
    """
    message = [{"role": "user", "content":prom}] + [{"role": "assistant", "content": resp.replace(" ки","")}]
    return rm_tokenizer.apply_chat_template(message, tokenize=False).replace(rm_tokenizer.bos_token, "")


data = []
z = 0
# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        #if len(sample["responses"]) < script_args.K:
        #    continue
        #test_texts = [change_of_format(sample['prompt'], tmp_output) for tmp_output in sample['answers']]
        test_texts = sample['my_solu'][0].replace("<|start_header_id|>user<|end_header_id|>\n\nYour most recent response is correct. Thanks.ENDSIGNAL\nReach max function call limit.", "")
        #all_len = [len(tmp_output) for tmp_output in sample['answers']]
        if z == 0:
            print(test_texts[0])
            z += 1
        rewards = get_reward(test_texts)
        data.append({"idx": sample["idx"], "gt": sample["gt"], "my_solu": sample["my_solu"], "preds": sample["preds"], "prox_rewards": rewards, "true_rewards": sample['rewards']})
# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)


if local_rank == 0:
    with open(script_args.output_dir, "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write('\n')

    dict_data = {
    "prompt": [d['prompt'] for d in gathered_data],
    "answers": [d['answers'] for d in gathered_data],
    "rewards": [d['rewards'] for d in gathered_data],
        "label": [d['label'] for d in gathered_data],
    }
    
    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_dict(dict_data)
    dataset = dataset.shuffle(seed=42)
    DatasetDict({"train": dataset}).push_to_hub(output_name)
