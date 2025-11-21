from typing import Any, Literal, List

import torch

from interfaces import BaseGenerator, BasePRM

import random
import numpy as np
import os
import re
import logging
import torch.nn.functional as F

from collections import defaultdict

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


class SelfConsistencySearch:
    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,

        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", 'prod'] = "min",
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

    def compute_step_scores(self, responses: list, prm_state):
        score_tok = getattr(self.prm, "score_token", None)
        responses = [r.replace("\n\n## Step", f"{score_tok}## Step") for r in responses]
        return self.prm(responses, prm_state, return_all_steps=self.return_all_steps)

    # def _update_temperature(self):
    #     if self.temp_update_rule is None:
    #         return None
    #     else:
    #         # TODO
    #         # self.generator.temperature = ...
    #         raise NotImplementedError()
        

    def _update_temperature(self):
        """
        Math100용 온도 스케줄:
        - 질문 하나 안에서 trial이 진행될수록: 탐색 → 활용 쪽으로 점진적으로 온도 감소
        - 최근 PRM aggregate score가 개선되면: 온도 살짝 낮춰서 더 '집중'
        - 최근 score가 악화되면: 온도 살짝 올려서 '다른 해법' 탐색
        """
        import math

        # generator에 temperature가 없으면 아무 것도 안 함
        if not hasattr(self.generator, "temperature"):
            return None

        # __call__ 초반에 세팅해둔 기준 온도 (없으면 현재 온도 사용)
        base_T = float(getattr(self, "_base_temperature", float(self.generator.temperature)))

        # trial 진행 비율 (0.0 ~ 1.0)
        # self.trials: 지금까지 시도한 횟수 (현재 trial 포함하도록 설계되어 있음)
        if self.max_trials is None or self.max_trials <= 1:
            progress = 0.0
        else:
            progress = max(0.0, min(1.0, self.trials / float(self.max_trials - 1)))

        # ── 1단계: 단순 스케줄 (초반엔 높은 온도, 후반으로 갈수록 낮은 온도)
        # 예: base_T 기준으로 처음에는 1.4배, 마지막에는 0.7배 정도로 수렴
        T_high = base_T * 1.4
        T_low = base_T * 0.7
        T_sched = (1.0 - progress) * T_high + progress * T_low

        # ── 2단계: 최근 PRM aggregate score 기반 미세조정
        scores = getattr(self, "_agg_scores", [])

        T = T_sched
        if len(scores) >= 2:
            last_score = float(scores[-1])
            best_prev = float(max(scores[:-1]))

            # diff > 0 이면 지금이 이전 최고보다 좋아진 것, diff < 0이면 나빠진 것
            diff = last_score - best_prev

            # Math100 특성상 스코어 스케일이 너무 민감하지 않도록 작은 임계값 사용
            tol = 0.02  # 필요하면 이 값만 조절해도 됨

            if diff > tol:
                # 최근 시도가 이전 최고보다 꽤 나아졌으면 -> 온도 약간 낮춰서 exploit
                T *= math.exp(-0.15)
            elif diff < -tol:
                # 최근 시도가 이전보다 떨어졌으면 -> 온도 약간 올려서 explore
                T *= math.exp(0.15)

        # 안전 범위 클램프
        T = max(0.2, min(1.8, T))

        # 실제 generator에 반영
        self.generator.temperature = float(T)

        logger.info(
            f"[TempUpdate-Math100] trials={self.trials}/{self.max_trials}, "
            f"base_T={base_T:.3f}, progress={progress:.2f}, "
            f"T_sched={T_sched:.3f}, final_T={T:.3f}, "
            f"scores={scores[-3:] if len(scores) >= 3 else scores}"
        )

        return T

    def __call__(self, question: str):
        # ─── 온도 초기화 (질문마다 독립적인 초기 T 사용) ───
        if hasattr(self.generator, "temperature"):
            # 첫 호출에서 기준 온도를 저장해두고, 이후엔 그걸로 리셋
            if not hasattr(self, "_base_temperature"):
                self._base_temperature = float(self.generator.temperature)
            self.generator.temperature = float(self._base_temperature)

        # 온도 업데이트에 쓸 히스토리 버퍼 초기화
        self._answers = []
        self._agg_scores = []
        
        input_ids_question = self.generator.encode(question)
        gen_state_question = self.generator.init_state(input_ids_question)
        prm_state_question = self.prm.init_state(question)

        input_ids_question = input_ids_question.repeat(self.init_number_of_beams, 1)
        gen_state_question = self.generator.inflate_state(gen_state_question, self.init_number_of_beams)
        prm_state_question = self.prm.inflate_state(prm_state_question, self.init_number_of_beams)

        input_len = input_ids_question.shape[1]
        complete_beams = defaultdict(list)
        
        proposal_ids, proposal_logits, gen_state = self.generator(input_ids_question, gen_state_question)
        self.trials += 1

        proposal_response_ids = proposal_ids[:, input_len :]
        proposal_response_text = self.generator.tokenizer.batch_decode(proposal_response_ids)
      
        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(proposal_response_text, prm_state_question)

        proposal_agg_scores = aggregate(proposal_scores, self.score_aggregation).item()

        # 첫 번째 후보도 히스토리에 기록
        self._answers.append(proposal_response_text[0])
        self._agg_scores.append(proposal_agg_scores)

        is_complete = self.generator.is_complete(proposal_ids)
        if not is_complete[0]:
            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'] = []
            complete_beams['aggregate_scores'] = []
            complete_beams['step_scores'] = []
            complete_beams['temp'] = [self.generator.temperature]
        else:
            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(proposal_response_text[0])
            complete_beams['aggregate_scores'].append(proposal_agg_scores)
            complete_beams['step_scores'].append(proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        self.trials = 0
        last_proposal_ids = proposal_ids.clone()
        best_score = proposal_agg_scores
        logger.info(f'[SelfConsistency] Intial {self.trials}/{self.max_trials} : {best_score:.4f}')

        for trial_idx in range(self.max_trials-1):
            self._update_temperature()
            new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(input_ids_question, gen_state_question)
            self.trials += 1

            new_proposal_respose_ids = new_proposal_ids[:, input_len :]

            new_proposal_response_text = self.generator.tokenizer.batch_decode(new_proposal_respose_ids)
            new_proposal_scores, new_proposal_score_logits, prm_state = self.compute_step_scores(new_proposal_response_text, prm_state_question)
            
            new_proposal_agg_scores = aggregate(new_proposal_scores, self.score_aggregation).item()
            logger.info(f'[SelfConsistency] New score {self.trials}/{self.max_trials} : {new_proposal_agg_scores:.4f}')

            # 새 후보도 히스토리에 기록 (온도 업데이트에 사용)
            self._answers.append(new_proposal_response_text[0])
            self._agg_scores.append(new_proposal_agg_scores)

            complete_beams['CaseType'].append('Candidates')
            complete_beams['answer'].append(new_proposal_response_text[0])
            complete_beams['aggregate_scores'].append(new_proposal_agg_scores)
            complete_beams['step_scores'].append(new_proposal_scores.tolist())
            complete_beams['temp'].append(self.generator.temperature)

        return complete_beams