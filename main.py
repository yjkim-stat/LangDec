import os
import json
import yaml
import torch
import random
import logging
import argparse
import numpy as np
from enum import Enum
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import is_dataclass, asdict

from datasets import load_dataset
import torch.backends.cudnn as cudnn

from general_prm import GeneralPRM
from deepseek_prm import DeepseekPRM
from llama_generator import LlamaGenerator
from search_sc import SelfConsistencySearch
from search_genetic import GeneticSearch

# Create a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Decoding using Hugging Face Transformers")

# Setup
parser.add_argument('--seed', type=int, default=123)
parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for authentication.")
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--version", type=str, default=None)

parser.add_argument('--dataset', type=str)
parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of new tokens to generate.")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of the model to use.")
parser.add_argument("--use_past_key_values", type=bool, default=False, help="Whether to use past key values for faster inference.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing.")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model (e.g., cuda, cpu).")
parser.add_argument("--secondary_device", type=str, default="cpu", help="Secondary device to offload computation (e.g., cpu).")

# Search config
parser.add_argument("--max_trials", type=int, default=None)
parser.add_argument("--pool_size", type=int, default=None)

parser.add_argument("--temp_change_rate", type=float, default=0.1)
parser.add_argument("--temp_floor", type=float, default=0.8)
parser.add_argument("--temp_ceil", type=float, default=1.2)

parser.add_argument("--score_aggregation", type=str, default=None)
parser.add_argument("--temp_update_rule", type=str, default=None)

parser.add_argument("--metric", type=str, default='top1')
parser.add_argument("--select_strategy", type=str, default='random')

# PRM config
parser.add_argument("--prm_model_name", type=str, default="UW-Madison-Lee-Lab/VersaPRM", help="Name of the model to use.")
parser.add_argument("--positive_tag", type=str, default="+", help="Positive tag used in the model.")
parser.add_argument("--negative_tag", type=str, default="-", help="Negative tag used in the model.")
parser.add_argument("--score_token", type=str, default=" \n\n\n\n", help="Token used to calculate or indicate scores.")

parser.add_argument(
    '--test_sample_idx',
    type=int,
    nargs='+',  # 여러 값 허용
    default=[],
    help="실행할 샘플 인덱스 리스트 (예: --test_sample_idx 1 5 7)"
)

logging.basicConfig(level=logging.INFO)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


dataset_query_key = {
    'openai/gsm8k' : 'question',
    'Idavidrein/gpqa' : 'Question',
    'Maxwell-Jia/AIME_2024' : 'Problem',
    'HuggingFaceH4/MATH-500': 'problem',
    'deepmind/aqua_rat' : 'question',
    'ChilleD/SVAMP' : 'question_concat',
    'amphora/MCLM': 'en'
}   

def to_jsonable(obj):
    # 1) PyTorch
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    # 2) NumPy
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # 3) 표준 컨테이너
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    # 4) dataclass, Enum 등
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    # 5) NaN/Inf 처리(옵션)
    if isinstance(obj, float):
        if obj != obj or obj in (float('inf'), float('-inf')):
            return None  # 혹은 문자열로 변환: str(obj)
    # 6) 기본형
    return obj

def save_config_and_prepare_dir(args):
    # 1) collect env vars
    env_vars = {
        key: os.environ.get(key)
        for key in ("CUDA_VISIBLE_DEVICES", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        if key in os.environ
    }

    # 2) build full config dict
    cfg = vars(args).copy()
    cfg.update(os.environ)

    if args.dataset == 'HuggingFaceH4/MATH-500':
        pretty_dataset = 'math500'
    elif args.dataset == 'TIGER-Lab/MMLU-Pro':
        pretty_dataset = 'mmlupro'
    elif args.dataset == 'amphora/MCLM':
        pretty_dataset = 'math100'
    else:
        raise KeyError()

    # 3) make a timestamped folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if os.environ.get('GDRIVE_DIR', False):
        base = Path(f"{os.environ.get('GDRIVE_DIR')}/experiments-{pretty_dataset}")
    else:
        base = Path(f"experiments-{pretty_dataset}")
    run_dir = base / f'{args.model_name.replace("/", "_")}-{args.prm_model_name.replace("/", "_")}-{args.method.replace("/", "_")}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 4) dump YAML
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return run_dir


if __name__ == '__main__':
    args = parser.parse_args()

    os.environ['ALGORITHM_VERSION'] = str(args.version)
    set_seed(args.seed)
    torch.set_num_threads(8)  # 사용 가능한 CPU 코어 수로 설정
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    bnb_config = {
        'load_in_4bit': True,
        'load_in_8bit': False,
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_compute_dtype': 'float16',
        'bnb_4bit_quant_type': 'nf4',
    }

    if args.dataset == 'openai/gsm8k':
        dataset = load_dataset(args.dataset, 'main', cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test'] 
    elif args.dataset == 'Idavidrein/gpqa':
        dataset = load_dataset(args.dataset, 'gpqa_diamond', cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['train'] 
    elif args.dataset == 'Maxwell-Jia/AIME_2024':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['train'] 
    elif args.dataset == 'HuggingFaceH4/MATH-500':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'deepmind/aqua_rat':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'ChilleD/SVAMP':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']     
    elif args.dataset == 'TIGER-Lab/MMLU-Pro':
        dataset = load_dataset(args.dataset, cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']
    elif args.dataset == 'amphora/MCLM':
        dataset = load_dataset(args.dataset, "MT-MATH100", cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']
    elif 'cais/mmlu' in args.dataset:
        assert '-' in args.dataset
        dataset = load_dataset('cais/mmlu', args.dataset.split('-')[-1], cache_dir=os.getenv('CACHE_DIR'))
        test_dataset = dataset['test']
    else:
        raise KeyError()

    if 'RLHFlow' in args.prm_model_name:
        prm = DeepseekPRM(
            quantization_config=bnb_config,
            model_name=args.prm_model_name,
            positive_tag=args.positive_tag,
            negative_tag=args.negative_tag,
            score_token=args.score_token,
            hf_token=args.hf_token,
            use_past_key_values=args.use_past_key_values,
            batch_size=args.batch_size,
            device=args.device,
            secondary_device=args.secondary_device,
            dtype=torch.float32,
        )        
    else:
        prm = GeneralPRM(
            quantization_config=bnb_config,
            model_name=args.prm_model_name,
            positive_tag=args.positive_tag,
            negative_tag=args.negative_tag,
            score_token=args.score_token,
            hf_token=args.hf_token,
            use_past_key_values=args.use_past_key_values,
            batch_size=args.batch_size,
            device=args.device,
            secondary_device=args.secondary_device,
            dtype=torch.float32,
        )

    if 'Llama' in args.model_name:
        generator = LlamaGenerator(
            max_new_tokens=args.max_new_tokens,
            model_name=args.model_name,
            quantization_config=bnb_config,
            hf_token=args.hf_token,
            use_past_key_values=args.use_past_key_values,
            batch_size=args.batch_size,
            device=args.device,
            secondary_device=args.secondary_device,
        )
    else:
        raise KeyError()

    if 'SC' in args.method:
        search = SelfConsistencySearch(
            method=args.method,
            generator = generator, 
            prm = prm,
            max_trials=args.max_trials,
            temp_update_rule=args.temp_update_rule,
            score_aggregation=args.score_aggregation,
        )
    elif 'Genetic' in args.method:
        search = GeneticSearch(
            method=args.method,
            generator = generator, 
            prm = prm,
            max_trials=args.max_trials,
            temp_update_rule=args.temp_update_rule,
            score_aggregation=args.score_aggregation,
            metric=args.metric,
            select_strategy=args.select_strategy,
        )        
        
    else:
        raise KeyError()
            
    past_input_tokens = 0
    past_output_tokens = 0
    results = defaultdict(dict)
    indices_to_run = range(len(test_dataset))
    if args.test_sample_idx:  # 하나라도 있으면
        indices_to_run = args.test_sample_idx

    for test_idx in tqdm(indices_to_run):
        test_sample = test_dataset[test_idx]
        if args.dataset == 'HuggingFaceH4/MATH-500':
            qid = test_idx

            # query 만들기question
            test_query = test_sample["problem"]
            formatted_query = f"{test_query}"
        elif args.dataset == 'amphora/MCLM':
            qid = test_idx

            # query 만들기question
            test_query = test_sample["en"]
            formatted_query = f"{test_query}"
        elif args.dataset == 'TIGER-Lab/MMLU-Pro':
            qid = test_sample["question_id"]  # MMLU-Pro 전용

            # query 만들기
            test_query = test_sample["question"]
            options = test_sample["options"]
            formatted_query = f"{test_query}\n"
            for i, choice in enumerate(options):
                formatted_query += f"{i}. {choice}\n"
        else:
            raise KeyError()

        try:
            outputs = search(formatted_query)
            result = {qid: outputs}

            # 개별 저장
            run_dir = save_config_and_prepare_dir(args)
            fname = f"qid{qid}.json"
            with open(run_dir /fname, "w") as f:
                json.dump(to_jsonable(result), f, indent=4)

        except Exception as err:
            import traceback
            print(traceback.format_exc() )
            
            logger.error(f"Error on qid {qid}: {err}")
            continue
