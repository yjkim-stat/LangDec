from typing import Any, Literal, List
from datetime import datetime

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
if os.getenv("DEBUG", False):
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
        raise NotImplementedError(f"{agg_method} aggregation is not implemented.")
    return aggregate_scores


class SelfConsistencySearch:
    """
    SelfConsistencySearch + TURN temperature selection.

    - 한 문제(question) 단위로:
      1) 짧은 probe trajectory 를 여러 온도(T_grid)에서 샘플링 (Budget과 별도)
      2) 각 온도별 mean entropy H(T)를 직접 계산
      3) ENTP (entropy phase transition point) 근사값으로 최적 temperature 추정
      4) 그 temperature 로 Self-Consistency (N회) 샘플링 수행

    - 현재 구현에서는:
      * 질문마다 TURN-probe 를 항상 새로 수행한다.
      * probe 결과로 얻어진 new_temp 를 base_temperature 로 저장하여,
        다음 질문의 시작 온도로 재사용한다.
    """

    def __init__(
        self,
        method,
        generator: BaseGenerator,
        prm: BasePRM,
        temp_update_rule=None,
        max_trials: int = None,
        score_aggregation: Literal["min", "mean", "last", "prod"] = "min",
        # TURN temperature grid step (주로 로깅용; 실제 grid는 turn_sample_size로 linspace)
        temp_change_rate: float = 0.1,
        # TURN 탐색 구간 [temp_floor, temp_ceil]
        temp_floor: float = 0.1,
        temp_ceil: float = 1.4,
        # TURN 에 사용할 probe 샘플 총 개수 (질문당 probe N회)
        turn_sample_size: int = 20,
        # (선택적) 온도 업데이트를 점진적으로 제한하고 싶을 때 사용하는 최대 이동량
        max_temp_change: float = 0.1,
        # ====== TURN probe 관련 옵션 ======
        # probe 샘플에서 사용할 최대 토큰 수 (full CoT보다 훨씬 작게)
        turn_probe_tokens: int = 32,
        # probe 를 몇 번 돌릴지 (실제 사용 샘플 수; turn_sample_size 보다 작게 써도 됨)
        turn_probe_trials: int = None,
    ):
        self.method = method
        self.generator = generator
        self.prm = prm

        self.temp_update_rule = temp_update_rule
        self.temp_change_rate = temp_change_rate
        self.temp_floor = temp_floor
        self.temp_ceil = temp_ceil
        self.turn_sample_size = turn_sample_size
        self.max_trials = max_trials
        self.trials = 0

        # generator 에 초기 temperature 가 있으면 그 값을 기본으로 사용
        if getattr(self.generator, "temperature", None) is not None:
            self.base_temperature = float(self.generator.temperature)
        else:
            self.base_temperature = 1.0
            self.generator.temperature = self.base_temperature

        # per-update 최대 이동량 설정 (gradual 모드에서 사용 가능)
        self.max_temp_change = max_temp_change
        self.score_aggregation = score_aggregation
        self.return_all_steps = True
        self.init_number_of_beams = 1

        # TURN 알고리즘용 logits 버퍼 (probe 에서 쌓은 샘플 – 디버깅용)
        self._turn_logits_buffer: List[torch.Tensor] = []

        # 질문 단위 TURN 상태
        self.turn_jump_once = True
        self.turn_refine = False
        self.turn_refine_span = 0.2   # 예: T* ± 0.2
        self.turn_refine_step = 0.02  # 국소 탐색 시 ΔT
        self._turn_refine_done = False
        self._last_predicted_temp = None

        # probe 설정
        self.turn_probe_tokens = int(turn_probe_tokens)
        # probe trial 횟수가 명시되지 않으면 turn_sample_size 와 동일하게 사용
        self.turn_probe_trials = int(turn_probe_trials) if turn_probe_trials is not None else int(turn_sample_size)

        # TURN에서 사용할 온도 격자:
        # temp_floor ~ temp_ceil 구간을 turn_sample_size 개로 균등 분할
        self.turn_temps = np.linspace(self.temp_floor, self.temp_ceil, self.turn_sample_size, dtype=float)
        num_temp_grid = len(self.turn_temps)

        logger.info(
            f"[Init] SelfConsistencySearch("
            f"method={self.method}, "
            f"base_temperature={self.base_temperature:.3f}, "
            f"temp_floor={self.temp_floor}, temp_ceil={self.temp_ceil}, "
            f"temp_change_rate={self.temp_change_rate}, "
            f"num_temp_grid={num_temp_grid}, "
            f"turn_sample_size={self.turn_sample_size}, "
            f"turn_probe_tokens={self.turn_probe_tokens}, "
            f"turn_probe_trials={self.turn_probe_trials}, "
            f"max_temp_change={self.max_temp_change})"
        )

    def compute_step_scores(self, responses: list, prm_state):
        score_tok = getattr(self.prm, "score_token", None)
        responses = [r.replace("\n\n## Step", f"{score_tok}## Step") for r in responses]
        return self.prm(responses, prm_state, return_all_steps=self.return_all_steps)

    @staticmethod
    def _compute_mean_entropy(logits: torch.Tensor) -> float:
        """
        [B, L, V] logits 에 대해 평균 토큰 엔트로피를 계산하는 보조 함수.
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = (-probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
        return entropy.mean().item()

    # -------------------------------
    # 1D moving average smoothing
    # -------------------------------
    @staticmethod
    def _smooth_1d(x: np.ndarray, window: int = 3) -> np.ndarray:
        """
        간단한 이동 평균 smoothing.
        - window 는 홀수 권장.
        - 데이터 길이가 짧을 때는 자동으로 줄임.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        if window <= 1 or n <= 2:
            return x.copy()

        # window 를 데이터 길이에 맞게 조정 (홀수 유지)
        if window > n:
            window = n if n % 2 == 1 else n - 1
        if window < 3:
            return x.copy()
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return x.copy()

        pad = window // 2
        # 양 끝을 edge 값으로 패딩
        x_pad = np.pad(x, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=float) / float(window)
        smoothed = np.convolve(x_pad, kernel, mode="valid")
        return smoothed

    # ---------------------------------------------------------------------
    # TURN: (temps, mean_entropy) 곡선을 이용해
    #       ENTP 근사값을 통해 최적 temperature 를 선택한다.
    # ---------------------------------------------------------------------
    def _turn_temperature(
        self,
        temps: np.ndarray,
        mean_entropy: np.ndarray,
        current_temperature: float,
        temp_floor: float = None,
        temp_ceil: float = None,
        allow_large_step: bool = False,
    ):
        """
        temps:
            - 길이 J 의 numpy array, 각 원소는 probe 시 실제로 사용한 temperature
        mean_entropy:
            - 길이 J 의 numpy array, 각 T 에서 측정된 mean token entropy H(T)
        current_temperature:
            - probe 이전의 base temperature (raw_delta 계산용)

        temp_floor, temp_ceil:
            - None 인 경우 self.temp_floor / self.temp_ceil 사용

        allow_large_step:
            - True  : predicted_temp 로 직접 점프
            - False : max_temp_change 를 이용해 한 스텝씩만 이동
        """
        if temps is None or mean_entropy is None or len(temps) == 0:
            logger.warning("[TempUpdate][TURN] empty temps/mean_entropy; skip update.")
            return None, None

        if temp_floor is None:
            temp_floor = self.temp_floor
        if temp_ceil is None:
            temp_ceil = self.temp_ceil

        # 온도 격자 정리 (오름차순 정렬)
        order = np.argsort(temps)
        temps = temps[order]
        mean_entropy = mean_entropy[order]
        num_temp_grid = len(temps)

        if num_temp_grid < 2:
            logger.warning("[TempUpdate][TURN] temperature grid too small; skip update.")
            return None, None

        original_temp = float(current_temperature)

        # 원본 log entropy
        log_entropy_raw = np.log(mean_entropy + 1e-12)
        # 스무딩된 log entropy (작은 진동 제거)
        log_entropy = self._smooth_1d(log_entropy_raw, window=3)

        # 로그 출력: 각 온도에서의 엔트로피 곡선 (raw + smooth)
        logger.info(
            "[TempUpdate][TURN] entropy_curve="
            f"{{'temps': {temps.tolist()}, "
            f"'mean_entropy': {mean_entropy.tolist()}, "
            f"'log_entropy_raw': {[float(v) for v in log_entropy_raw]}, "
            f"'log_entropy_smooth': {[float(v) for v in log_entropy]}, "
            f"'num_points': {num_temp_grid}}}"
        )

        # 2) 2차 미분으로 concave/convex 분석 (스무딩된 log_entropy 기준)
        second_derivative = np.gradient(
            np.gradient(log_entropy, temps), temps
        )

        logger.info(
            f"[TempUpdate][TURN] second_derivative="
            f"{[float(v) for v in second_derivative]}"
        )

        # ------------------------------------------------------------------
        # (A) 2차 미분 sign-change 기반 ENTP:
        #     d^2/dT^2 log H(T)가 음수→양수로 바뀌는 지점들 중
        #     "평균 엔트로피가 가장 낮은 지점"을 선택해서 단일 ENTP 로 사용.
        # ------------------------------------------------------------------
        j_star = None
        mode = "unknown"

        sign_change = np.where(
            (second_derivative[1:] > 0) & (second_derivative[:-1] <= 0)
        )[0]  # 실제 인덱스는 +1 필요

        if len(sign_change) > 0:
            candidates = sign_change + 1  # 0-based offset 보정

            # 후보들의 mean_entropy 및 second_derivative 를 같이 로깅
            cand_info = [
                {
                    "idx": int(j),
                    "temp": float(temps[j]),
                    "mean_entropy": float(mean_entropy[j]),
                    "second_derivative": float(second_derivative[j]),
                }
                for j in candidates
            ]
            logger.info(
                f"[TempUpdate][TURN] sign-change candidates={cand_info}"
            )

            # 여러 candidate 중에서 "평균 엔트로피가 가장 낮은 지점"을 선택
            cand_entropies = mean_entropy[candidates]
            best_local = int(np.argmin(cand_entropies))
            best_idx = int(candidates[best_local])

            j_star = best_idx
            mode = "sign-change-min-entropy"
            logger.info(
                f"[TempUpdate][TURN] mode='{mode}', "
                f"chosen_j_star={j_star}, "
                f"T_grid[j_star]={float(temps[j_star]):.3f}, "
                f"chosen_mean_entropy={float(mean_entropy[j_star]):.6f}, "
                f"chosen_second_derivative={float(second_derivative[j_star]):.6f}"
            )

        # ------------------------------------------------------------------
        # (B) sign-change 가 전혀 없는 경우 (전체 concave 등):
        #     Kneedle 스타일 knee 탐색 (끝점을 잇는 직선과의 거리 최대 지점).
        # ------------------------------------------------------------------
        if j_star is None:
            x = temps
            y = log_entropy

            denom = (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 1.0
            slope = (y[-1] - y[0]) / denom
            line = y[0] + slope * (x - x[0])

            diff = y - line  # concave 이면 가운데에서 diff > 0

            logger.info(
                "[TempUpdate][TURN] knee-debug="
                f"{{'line_first': {float(line[0]):.6f}, "
                f"'line_last': {float(line[-1]):.6f}, "
                f"'diff': {[float(v) for v in diff]}}}"
            )

            if len(x) > 2:
                diff_inner = diff[1:-1]
                if np.all(np.isfinite(diff_inner)) and not np.all(
                    np.isclose(diff_inner, diff_inner[0])
                ):
                    j_knee_inner = int(np.argmax(np.abs(diff_inner)))
                    j_star = 1 + j_knee_inner  # 내부 offset 보정
                    mode = "knee-fallback"
                    logger.info(
                        f"[TempUpdate][TURN] mode='{mode}', j_star={j_star}, "
                        f"T_grid[j_star]={float(temps[j_star]):.3f}, "
                        f"max_abs_diff={float(np.max(np.abs(diff_inner))):.6f}"
                    )

        # ------------------------------------------------------------------
        # (C) 여전히 j_star 가 정해지지 않았다면 (거의 직선 등):
        #     log_entropy 최소 인덱스를 사용하되,
        #     그게 boundary 면 중앙값으로 교체.
        # ------------------------------------------------------------------
        if j_star is None:
            j_min = int(np.argmin(log_entropy))
            if j_min == 0 or j_min == len(temps) - 1:
                j_star = len(temps) // 2
                mode = "flat-fallback-center"
            else:
                j_star = j_min
                mode = "flat-fallback-min"
            logger.info(
                f"[TempUpdate][TURN] mode='{mode}', "
                f"j_min={j_min}, j_star={j_star}, "
                f"T_grid[j_star]={float(temps[j_star]):.3f}"
            )

        # 여기까지 오면 j_star 는 항상 [0, len(temps)-1] 범위의 유효한 인덱스
        t_star = float(temps[j_star])

        beta_a = 0.0  # majority voting / self-consistency 에서는 0.0 사용
        predicted_temp = float(
            np.clip(t_star + beta_a, temp_floor, temp_ceil)
        )

        raw_delta = predicted_temp - original_temp
        if allow_large_step:
            step = raw_delta
            new_temp = float(predicted_temp)
        else:
            max_step = float(self.max_temp_change)
            step = float(np.clip(raw_delta, -max_step, max_step))
            new_temp = float(original_temp + step)

        new_temp = float(np.clip(new_temp, temp_floor, temp_ceil))

        logger.info(
            f"[TempUpdate][TURN] result="
            f"{{'t_star': {t_star:.3f}, "
            f"'beta_a': {beta_a:.3f}, "
            f"'predicted_temp': {predicted_temp:.3f}, "
            f"'raw_delta': {raw_delta:.3f}, "
            f"'applied_step': {step:.3f}, "
            f"'new_temp': {new_temp:.3f}, "
            f"'mode': '{mode}'}}"
        )

        # 실제 generator 온도 갱신
        self.generator.temperature = new_temp

        return predicted_temp, new_temp

    # ---------------------------------------------------------------------
    # 질문 시작 시, 짧은 probe 로 TURN 을 수행해 temperature 를 한 번 결정
    # (probe 단계에서 temp 를 linspace grid 값으로 actual 샘플링)
    # ---------------------------------------------------------------------
    def _run_turn_probe(
        self,
        input_ids_question: torch.Tensor,
        gen_state_question: Any,
    ):
        """
        질문당 한 번만, TURN 전용 probe 를 수행한다.

        - self.turn_temps (temp_floor~temp_ceil, turn_sample_size개)를 돌면서,
          generator.temperature = T 로 실제 샘플링.
        - 전체 probe 예산(turn_probe_trials) 안에서
          각 온도별로 최대 1개 샘플을 확보 (선형 스캔).
        - 각 T 에서의 mean entropy 를 직접 계산한 뒤,
          (T, H(T)) 곡선으로 ENTP 기반 temperature 를 선택한다.
        """

        original_max_new_tokens = getattr(self.generator, "max_new_tokens", None)
        original_temp = float(self.generator.temperature)

        logger.info(
            f"[TURN-Probe] start: "
            f"base_temperature(before probe)={original_temp:.3f}, "
            f"probe_tokens={self.turn_probe_tokens}, "
            f"probe_trials={self.turn_probe_trials}"
        )

        # generator 의 max_new_tokens 를 짧게 줄여서 probe 수행
        if original_max_new_tokens is not None:
            self.generator.max_new_tokens = int(self.turn_probe_tokens)

        # logits 버퍼 및 probe entropy 통계 초기화
        self._turn_logits_buffer.clear()

        temps_grid = self.turn_temps
        num_temp_grid = len(temps_grid)

        # 전체 probe 예산: turn_probe_trials 와 temp grid 개수 중 작은 쪽
        num_probes_total = min(self.turn_probe_trials, num_temp_grid)
        if num_probes_total < num_temp_grid:
            logger.warning(
                "[TURN-Probe] probe budget (%d) < num_temp_grid (%d); "
                "일부 온도만 샘플링",
                num_probes_total,
                num_temp_grid,
            )

        # 온도별 entropy 리스트
        temp_to_entropies = {float(t): [] for t in temps_grid}

        # 각 온도에 대해 최대 1회씩 probe 수행 (선형 스캔)
        for p_idx in range(num_probes_total):
            t = float(temps_grid[p_idx])
            self.generator.temperature = t

            proposal_ids, proposal_logits, _ = self.generator(
                input_ids_question, gen_state_question
            )
            self._turn_logits_buffer.append(proposal_logits)

            entropy_val = self._compute_mean_entropy(proposal_logits)
            temp_to_entropies[t].append(entropy_val)

            B, L_i, V = proposal_logits.shape
            num_tokens = B * L_i

            # logger.info(
            #     f"[TURN-Probe] collected logits sample {p_idx+1}/{num_probes_total} "
            #     f"(temp={t:.3f}, mean_entropy={entropy_val:.6f}, num_tokens={num_tokens})"
            # )

        # 각 온도별 엔트로피 통계 및 TURN 입력 곡선 생성
        temps_for_curve = []
        mean_entropy_for_curve = []
        probe_entropy_all = []

        for t in temps_grid:
            ent_list = temp_to_entropies[float(t)]
            if len(ent_list) == 0:
                continue
            ent_array = np.array(ent_list, dtype=float)
            temps_for_curve.append(float(t))
            mean_entropy_for_curve.append(float(ent_array.mean()))
            probe_entropy_all.extend(ent_list)

            # logger.info(
            #     "[TURN-Probe] temp=%.3f, entropy_stats={'count': %d, 'mean': %.6f, 'min': %.6f, 'max': %.6f}",
            #     float(t),
            #     len(ent_list),
            #     ent_array.mean(),
            #     ent_array.min(),
            #     ent_array.max(),
            # )

        if len(probe_entropy_all) > 0:
            probe_entropy_all_np = np.array(probe_entropy_all, dtype=float)
            logger.info(
                "[TURN-Probe] probe_entropy_stats_all="
                f"{{'count': {len(probe_entropy_all)}, "
                f"'mean': {probe_entropy_all_np.mean():.6f}, "
                f"'min': {probe_entropy_all_np.min():.6f}, "
                f"'max': {probe_entropy_all_np.max():.6f}}}"
            )

        temps_for_curve = np.array(temps_for_curve, dtype=float)
        mean_entropy_for_curve = np.array(mean_entropy_for_curve, dtype=float)

        if len(temps_for_curve) == 0:
            logger.warning("[TURN-Probe] no entropy data collected; skip TURN update.")
            # generator.temperature 는 original_temp 로 복원
            self.generator.temperature = original_temp
            if original_max_new_tokens is not None:
                self.generator.max_new_tokens = int(original_max_new_tokens)
            return

        # TURN 로 ENTP 기반 temperature 선택 (각기 다른 temp 에서 얻은 H(T) 곡선 사용)
        predicted_temp, new_temp = self._turn_temperature(
            temps_for_curve,
            mean_entropy_for_curve,
            current_temperature=original_temp,
            temp_floor=self.temp_floor,
            temp_ceil=self.temp_ceil,
            allow_large_step=True,  # probe 후에는 한번에 점프
        )
        self._last_predicted_temp = predicted_temp

        # base_temperature 를 새로 선택된 온도로 업데이트
        if new_temp is not None:
            self.base_temperature = float(new_temp)

        # generator의 max_new_tokens 복원
        if original_max_new_tokens is not None:
            self.generator.max_new_tokens = int(original_max_new_tokens)

        logger.info(
            f"[TURN-Probe] done: predicted_temp={predicted_temp}, "
            f"applied_temperature(now)={self.generator.temperature:.3f}, "
            f"base_temperature(for next question)={self.base_temperature:.3f}"
        )

    def __call__(self, question: str):
        # TURN 상태 리셋 (질문 단위 상태 초기화)
        self._turn_logits_buffer.clear()
        self.trials = 0
        self._turn_refine_done = False
        self._last_predicted_temp = None

        # 질문 시작 시 base_temperature 로 초기화
        self.generator.temperature = float(self.base_temperature)

        now = datetime.now()
        weekday_names = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        am_pm = "오전" if now.hour < 12 else "오후"
        hour_12 = now.hour % 12 or 12
        logger.info(
            f"{now:%Y%m%d} {weekday_names[now.weekday()]} {am_pm} {hour_12}시 {now.minute:02d}분 작업"
        )
        logger.info(
            f"[Temp][Question Start] temperature={float(self.generator.temperature):.3f}, "
            f"base_temperature={self.base_temperature:.3f}"
        )

        # 인코딩 / 상태 초기화
        input_ids_question = self.generator.encode(question)
        gen_state_question = self.generator.init_state(input_ids_question)
        prm_state_question = self.prm.init_state(question)

        input_ids_question = input_ids_question.repeat(self.init_number_of_beams, 1)
        gen_state_question = self.generator.inflate_state(
            gen_state_question, self.init_number_of_beams
        )
        prm_state_question = self.prm.inflate_state(
            prm_state_question, self.init_number_of_beams
        )

        input_len = input_ids_question.shape[1]
        complete_beams = defaultdict(list)

        # ============================================================
        # 1) TURN 전용 probe 단계: Budget(Trial)에 포함되지 않음
        # ============================================================
        if self.temp_update_rule == "turn":
            self._run_turn_probe(input_ids_question, gen_state_question)
        else:
            logger.info("[TURN-Probe] temp_update_rule is not 'turn'; skip probe.")

        current_temp = (
            float(self.generator.temperature)
            if self.generator.temperature is not None
            else float("nan")
        )
        logger.info(
            f"[Temp][Before Trial 0] temperature after TURN-probe: {current_temp:.3f}"
        )

        # ============================================================
        # 2) Self-Consistency Trial 0 (full reasoning)
        # ============================================================
        proposal_ids, proposal_logits, gen_state = self.generator(
            input_ids_question, gen_state_question
        )
        self.trials += 1

        proposal_response_ids = proposal_ids[:, input_len:]
        proposal_response_text = self.generator.tokenizer.batch_decode(
            proposal_response_ids
        )

        proposal_scores, proposal_score_logits, prm_state = self.compute_step_scores(
            proposal_response_text, prm_state_question
        )

        proposal_agg_scores = aggregate(
            proposal_scores, self.score_aggregation
        ).item()

        is_complete = self.generator.is_complete(proposal_ids)
        if not is_complete[0]:
            complete_beams["CaseType"].append("Candidates")
            complete_beams["answer"] = []
            complete_beams["aggregate_scores"] = []
            complete_beams["step_scores"] = []
            complete_beams["temp"] = [self.generator.temperature]
        else:
            complete_beams["CaseType"].append("Candidates")
            complete_beams["answer"].append(proposal_response_text[0])
            complete_beams["aggregate_scores"].append(proposal_agg_scores)
            complete_beams["step_scores"].append(proposal_scores.tolist())
            complete_beams["temp"].append(self.generator.temperature)

        best_score = proposal_agg_scores
        logger.info(
            f"[SelfConsistency] Initial {self.trials}/{self.max_trials} : {best_score:.4f}"
        )
        logger.info(
            f"[Temp][Trial 0] temperature after sampling: "
            f"{float(self.generator.temperature) if self.generator.temperature is not None else float('nan'):.3f}"
        )

        # ============================================================
        # 3) 나머지 Trial (1 ~ max_trials-1)
        #    - 현재 구현: TURN 재적용 없이 temperature 고정
        # ============================================================
        for trial_idx in range(self.max_trials - 1):
            current_temp = (
                float(self.generator.temperature)
                if self.generator.temperature is not None
                else float("nan")
            )
            logger.info(
                f"[Temp][Trial {trial_idx+1}] temperature before sampling: {current_temp:.3f}"
            )

            new_proposal_ids, new_proposal_logits, new_gen_state = self.generator(
                input_ids_question, gen_state_question
            )
            self.trials += 1

            new_proposal_response_ids = new_proposal_ids[:, input_len:]
            new_proposal_response_text = self.generator.tokenizer.batch_decode(
                new_proposal_response_ids
            )
            (
                new_proposal_scores,
                new_proposal_score_logits,
                prm_state,
            ) = self.compute_step_scores(
                new_proposal_response_text, prm_state_question
            )

            new_proposal_agg_scores = aggregate(
                new_proposal_scores, self.score_aggregation
            ).item()
            logger.info(
                f"[SelfConsistency] New score {self.trials}/{self.max_trials} : "
                f"{new_proposal_agg_scores:.4f}"
            )
            logger.info(
                f"[Temp][Trial {trial_idx+1}] temperature after sampling: "
                f"{float(self.generator.temperature) if self.generator.temperature is not None else float('nan'):.3f}"
            )

            complete_beams["CaseType"].append("Candidates")
            complete_beams["answer"].append(new_proposal_response_text[0])
            complete_beams["aggregate_scores"].append(new_proposal_agg_scores)
            complete_beams["step_scores"].append(new_proposal_scores.tolist())
            complete_beams["temp"].append(self.generator.temperature)

        return complete_beams
