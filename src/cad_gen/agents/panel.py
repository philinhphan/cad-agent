"""Critic panel + conservative aggregation.

Runs the critic across N samples and/or several models concurrently and folds the
results into one conservative verdict (lowest/median score, unioned issues, worst
checklist status per requirement) so a single lenient model cannot rescue a bad part.
With one model and one sample it degenerates to a single ``run_critique`` call.
"""

import asyncio
import statistics
from pathlib import Path

from pydantic_ai.models import Model

from cad_gen.agents.critic import build_critic_agent, run_critique
from cad_gen.models import (
    CheckReport,
    Critique,
    DrawingAttachment,
    DrawingTarget,
    ExecutionResult,
)

_STATUS_RANK = {"pass": 0, "uncertain": 1, "fail": 2}


async def run_critic_panel(
    *,
    models: list[str | Model],
    samples: int,
    aggregation: str,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
    drawings: list[DrawingAttachment] | None = None,
    target: DrawingTarget | None = None,
    check_report: CheckReport | None = None,
    section_path: Path | None = None,
) -> tuple[Critique, list[Critique]]:
    """Return (aggregated critique, raw panel critiques). Panel calls run concurrently."""
    agents = [build_critic_agent(m) for m in models]
    tasks = [
        run_critique(
            agent,
            spec=spec,
            execution=execution,
            render_path=render_path,
            drawings=drawings,
            target=target,
            check_report=check_report,
            section_path=section_path,
        )
        for agent in agents
        for _ in range(max(1, samples))
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    critiques = [r for r in results if isinstance(r, Critique)]
    if not critiques:
        raise next(r for r in results if isinstance(r, BaseException))
    return _aggregate(critiques, aggregation), critiques


def _aggregate(critiques: list[Critique], aggregation: str) -> Critique:
    if len(critiques) == 1:
        return critiques[0]
    scores = [c.score for c in critiques]
    score = (
        int(round(statistics.median(scores)))
        if aggregation == "median"
        else min(scores)
    )
    worst = min(critiques, key=lambda c: c.score)
    return Critique(
        matches_spec=all(c.matches_spec for c in critiques) and score >= 8,
        score=score,
        issues=_dedup(i for c in critiques for i in c.issues),
        suggestions=_dedup(s for c in critiques for s in c.suggestions),
        summary=worst.summary,
        checklist=_merge_checklist(critiques),
        dimensional_score=_min_sub(c.dimensional_score for c in critiques),
        feature_completeness_score=_min_sub(c.feature_completeness_score for c in critiques),
        proportion_score=_min_sub(c.proportion_score for c in critiques),
    )


def _dedup(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _min_sub(values):
    nums = [v for v in values if v is not None]
    return min(nums) if nums else None


def _merge_checklist(critiques: list[Critique]):
    by_req: dict[str, object] = {}
    for c in critiques:
        for item in c.checklist:
            key = item.requirement.strip().lower()
            cur = by_req.get(key)
            if cur is None or _STATUS_RANK[item.status] > _STATUS_RANK[cur.status]:
                by_req[key] = item
    return list(by_req.values())
