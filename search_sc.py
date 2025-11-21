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

    def _update_temperature(self):
        """
        Math100용 공격적인 쿨링 스케줄:
        - 각 question마다 __call__ 초반에 저장한 _base_temperature를 기준으로
          trial이 진행될수록 온도를 빠르게 낮춘다.
        - 초기엔 base_T에서 시작하고, 후반부에 가까워질수록
          대략 0.35 * base_T 수준까지 강하게 식힌다.
        - PRM 점수나 답의 다양성은 사용하지 않고,
          오로지 trial 진행 정도(progress)에만 의존한다.
        """
        import math

        # generator에 temperature가 없으면 아무 것도 하지 않음
        if not hasattr(self.generator, "temperature"):
            return None

        # __call__에서 한 번 저장해둔 기준 온도 (없으면 현재 온도 사용)
        base_T = float(getattr(self, "_base_temperature", float(self.generator.temperature)))

        # max_trials 정보가 없거나 1 이하이면 그냥 base_T 유지
        if self.max_trials is None or self.max_trials <= 1:
            self.generator.temperature = base_T
            return base_T

        # self.trials 는 for 루프 안에서:
        #   - _update_temperature() 호출 시점: 0, 1, 2, ... (0-based 인덱스 느낌)
        #   - 최대값은 (max_trials - 2) 정도
        step_idx = max(0, min(self.trials, self.max_trials - 1))
        progress = step_idx / float(self.max_trials - 1)  # 0.0 ~ 1.0

        # ── 더욱 공격적인 쿨링 파라미터 ──
        target_min_ratio = 0.35   # 마지막에는 base_T의 35% 근처까지 떨어뜨리기
        alpha = 2.0               # alpha > 1 이면 초반부터 빠르게 식음 (front-loaded cooling)

        # ratio(progress):
        #   progress = 0   → ratio = 1.0 (base_T)
        #   progress ~ 0.5 → ratio ≈ 0.51 (이미 꽤 낮음)
        #   progress → 1   → ratio = target_min_ratio (0.35)
        ratio = target_min_ratio + (1.0 - target_min_ratio) * ((1.0 - progress) ** alpha)

        T = base_T * ratio

        # 안전 범위 클램프 (너무 극단적이지 않도록)
        # 정확도에 올인하는 세팅이므로 상한은 1.0 정도로 두고,
        # 하한은 0.15까지 허용
        T = max(0.15, min(1.0, T))

        # 실제 generator에 반영
        self.generator.temperature = float(T)

        logger.info(
            f"[TempUpdate-Math100-AggressiveCooling] "
            f"trials={self.trials}/{self.max_trials}, "
            f"base_T={base_T:.3f}, progress={progress:.2f}, "
            f"ratio={ratio:.4f}, T={T:.3f}"
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