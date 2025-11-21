
import os
import re
import random
import logging
import numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from interfaces import BaseGenerator, BasePRM

level = logging.INFO
if os.getenv('DEBUG', False):
    level = logging.DEBUG

# 로깅 설정
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bootstrap_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def aggregate(vals, agg_method):
    if agg_method == "min":
        aggregate_scores, _ = torch.min(vals, dim=-1)
    elif agg_method == "mean":
        aggregate_scores = torch.mean(vals, dim=-1)
    elif agg_method == "sum":
        aggregate_scores = torch.sum(vals, dim=-1)
    elif agg_method == "last":
        aggregate_scores = vals[:, -1]
    elif agg_method == "prod":
        aggregate_scores = torch.cumprod(vals, dim=1)[:, -1]
    else:
        raise NotImplementedError(
            f"{agg_method} aggregation is not implemented."
        )
    return aggregate_scores


def select_case_index(
    complete_beams: dict,
    strategy: str = "random",   # "best", "random", 또는 정수 인덱스
) -> int:
    """
    (1) complete_beams에서 사용할 케이스 하나 선택하고, 그 인덱스를 리턴하는 함수.
    complete_beams["aggregate_scores"]를 기준으로 선택.
    """
    scores = complete_beams.get("aggregate_scores", None)
    if scores is None or len(scores) == 0:
        raise ValueError("complete_beams['aggregate_scores']가 비어 있습니다.")

    if strategy == "best":
        return int(np.argmax(scores))
    elif strategy == "random":
        return random.randint(0, len(scores)-1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def build_cur_input_ids_from_maxima(
    question: str,
    generator,
    complete_beams: dict,
    case_idx: int,
    maxima_mode: str = "local_max",   # "local_max" 또는 "global_max"
    step_pattern: str = r"## Step",
) -> torch.Tensor:
    """
    (2) 선택된 case_idx에 대해 step_scores에서 maxima를 찾고,
        그 지점까지의 answer를 prefix로 사용해서 cur_input_ids를 만들어 반환.

    반환: cur_input_ids (1, L)  — question + truncated_answer 전체를 encode한 결과
    """
    answers = complete_beams.get("answer", None)
    step_scores_list = complete_beams.get("step_scores", None)

    if answers is None or step_scores_list is None:
        raise ValueError("complete_beams에 'answer' 또는 'step_scores'가 없습니다.")

    answer = answers[case_idx]
    step_scores = torch.tensor(step_scores_list[case_idx], dtype=torch.float32)  # (T,)

    # maxima 위치 찾기
    if maxima_mode == "local_max":
        local_max_idxs = find_local_maxima(step_scores)
        if len(local_max_idxs) > 0:
            anchor_step = local_max_idxs[0]  # 가장 높은 local maxima
        else:
            # local maxima가 없으면 global max로 fallback
            anchor_step = int(torch.argmax(step_scores).item())
    elif maxima_mode == "global_max":
        anchor_step = int(torch.argmax(step_scores).item())
    else:
        raise ValueError(f"Unknown maxima_mode: {maxima_mode}")

    # anchor_step까지의 답변만 남기기
    truncated_answer = _truncate_answer_at_step(
        answer,
        step_index=anchor_step,
        step_pattern=step_pattern,
    )

    # question + truncated_answer를 하나의 프롬프트로 묶어서 encode
    # (원래 프롬프트 포맷에 맞게 "\n\n" 등은 원하는 대로 바꿔도 됨)
    full_prompt = question + "\n" + truncated_answer
    cur_input_ids = generator.encode(full_prompt)  # (1, L)

    return cur_input_ids

def compute_metric(
    metric: str,
    logits: torch.Tensor,           # (T, V)
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)  # (T,V)
    # 표준: entropy = -sum p log p; perplexity = exp(entropy)
    if metric in ("top1"):
        metric = probs.topk(1, dim=-1).values.sum(-1)
    else:
        raise KeyError(metric)

    return metric

def find_local_maxima(nll: torch.Tensor):
    T = nll.numel()
    idxs, vals = [], []
    for i in range(1, T-1):
        if (nll[i-1] < nll[i]) and (nll[i] >= nll[i+1]):
            idxs.append(i)
            vals.append(float(nll[i]))
    # 가장 높은 maxima의 index부터 전달하게 됩니다.
    return [i for i, _ in sorted(zip(idxs, vals), key=lambda x: -x[1])]

class GeneticSearch:
    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,

        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", 'prod'] = "min",
        metric:str = 'top1',
        select_strategy:str = 'random'
    ):
        self.method = method
        self.generator = generator
        self.prm = prm

        self.temp_update_rule = temp_update_rule
        self.max_trials = max_trials
        self.trials = 0

        self.score_aggregation = score_aggregation

        self.return_all_steps = True

        self.init_number_of_beams = 1

        self.trial_to_ids = {}
        self.trial_to_logits = {}
        self.metric = metric
        self.select_strategy = select_strategy

    def compute_step_scores(self, responses: list, prm_state):
        score_tok = getattr(self.prm, "score_token", None)
        responses = [r.replace("\n\n## Step", f"{score_tok}## Step") for r in responses]
        return self.prm(responses, prm_state, return_all_steps=self.return_all_steps)

    def _update_temperature(self):
        if self.temp_update_rule is None:
            return None
        else:
            # TODO
            # self.generator.temperature = ...
            raise NotImplementedError()

    def __call__(self, question: str):
        self.trial_to_ids = {}
        self.trial_to_logits = {}
        self.trials = 0
        input_ids_question = self.generator.encode(question)
        gen_state_question = self.generator.init_state(input_ids_question)
        prm_state_question = self.prm.init_state(question)

        input_ids_question = input_ids_question.repeat(self.init_number_of_beams, 1)
        gen_state_question = self.generator.inflate_state(gen_state_question, self.init_number_of_beams)
        prm_state_question = self.prm.inflate_state(prm_state_question, self.init_number_of_beams)

        input_len = input_ids_question.shape[1]
        complete_beams = defaultdict(list)
        
        proposal_ids, proposal_logits, gen_state = self.generator(input_ids_question, gen_state_question)
        # print(f'proposal_logits:{proposal_logits.shape}')
        # proposal_metric_seq = compute_metric(self.metric, proposal_logits)
        # peak_ids = find_local_maxima(proposal_metric_seq)
        self.trial_to_ids[self.trials] = proposal_ids
        self.trial_to_logits[self.trials] = proposal_logits
        self.trials += 1

        proposal_response_ids = proposal_ids[:, input_len :]
        proposal_response_text = self.generator.tokenizer.batch_decode(proposal_response_ids)
      
        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(proposal_response_text, prm_state_question)

        proposal_agg_scores = aggregate(proposal_scores, self.score_aggregation).item()

        is_complete = self.generator.is_complete(proposal_ids)
        # if not is_complete[0]:
        #     complete_beams['CaseType'].append('Candidates')
        #     complete_beams['answer'] = []
        #     complete_beams['aggregate_scores'] = []
        #     complete_beams['step_scores'] = []
        #     complete_beams['temp'] = [self.generator.temperature]
        # else:
        complete_beams['CaseType'].append('Candidates')
        complete_beams['answer'].append(proposal_response_text[0])
        complete_beams['aggregate_scores'].append(proposal_agg_scores)
        complete_beams['step_scores'].append(proposal_scores.tolist())
        complete_beams['temp'].append(self.generator.temperature)

        # last_proposal_ids = proposal_ids.clone()
        logger.info(f'[Genetic] Intial {self.trials}/{self.max_trials} : {proposal_agg_scores:.4f}')

        for trial_idx in range(self.max_trials-1):
            selected_idx = select_case_index(complete_beams, strategy=self.select_strategy)
            selected_ids = self.trial_to_ids[selected_idx]
            selected_logits = self.trial_to_logits[selected_idx][0]
            print(f'selected_logits:{selected_logits.shape}')
            proposal_metric_seq = compute_metric(self.metric, selected_logits)
            peak_ids = find_local_maxima(proposal_metric_seq)
            self._update_temperature()
            if len(peak_ids) == 0:
                # peak이 없는 상황의 경우 그냥 처음부터 새로 생성
                input_ids_for_proposal = input_ids_question
                new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_for_proposal, gen_state_question)
                self.trial_to_ids[self.trials] = new_proposal_ids
                self.trial_to_logits[self.trials] = new_proposal_logits
                self.trials += 1

                new_proposal_respose_ids = new_proposal_ids[:, input_len :]

                new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
                new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
                
                new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
                logger.info(f'[Genetic] Generating from question {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

                complete_beams['CaseType'].append('Candidates')
                complete_beams['answer'].append(new_proposal_response_text[0])
                complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
                complete_beams['step_scores'].append(new_proposal_scores.tolist())
                complete_beams['temp'].append(self.generator.temperature)                
            else:
                peak_idx = peak_ids[0] # 가장 높은 local maxima를 선택함
                input_ids_for_proposal = selected_ids[:, :input_len+peak_idx]
                new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_for_proposal, gen_state_question)
                self.trial_to_ids[self.trials] = new_proposal_ids
                self.trial_to_logits[self.trials] = new_proposal_logits
                self.trials += 1

                new_proposal_respose_ids = new_proposal_ids[:, input_len :]

                new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
                new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
                
                new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
                logger.info(f'[Genetic] Generating from {selected_idx}-th sample {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

                complete_beams['CaseType'].append('Candidates')
                complete_beams['answer'].append(new_proposal_response_text[0])
                complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
                complete_beams['step_scores'].append(new_proposal_scores.tolist())
                complete_beams['temp'].append(self.generator.temperature)

        return complete_beams