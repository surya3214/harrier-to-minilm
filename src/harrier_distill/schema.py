"""Example schema validation for distillation shards."""

from __future__ import annotations

from typing import Any, Iterable

REQUIRED_FIELDS = (
    "example_id",
    "task",
    "lang",
    "texts",
    "roles",
    "teacher_prompt_names",
    "student_prompt_names",
)

ALLOWED_TASKS = {"sts", "retrieval", "bitext", "clustering", "classification", "reranking"}
ALLOWED_ROLES = {"query", "positive", "negative", "document", "sentence_a", "sentence_b"}


class SchemaError(ValueError):
    pass


def validate_example(ex: dict[str, Any]) -> dict[str, Any]:
    for field in REQUIRED_FIELDS:
        if field not in ex:
            raise SchemaError(f"Missing required field: {field}")

    if ex["task"] not in ALLOWED_TASKS:
        raise SchemaError(f"Invalid task: {ex['task']}")

    texts = ex["texts"]
    roles = ex["roles"]
    t_prompts = ex["teacher_prompt_names"]
    s_prompts = ex["student_prompt_names"]

    if not isinstance(texts, list) or not texts:
        raise SchemaError("texts must be a non-empty list")
    if not (len(texts) == len(roles) == len(t_prompts) == len(s_prompts)):
        raise SchemaError(
            "texts, roles, teacher_prompt_names, student_prompt_names must have equal length"
        )
    for role in roles:
        if role not in ALLOWED_ROLES:
            raise SchemaError(f"Invalid role: {role}")
    for t in texts:
        if not isinstance(t, str) or not t.strip():
            raise SchemaError("All texts must be non-empty strings")

    # label is optional
    if "label" not in ex:
        ex = {**ex, "label": None}
    if "text_ids" not in ex:
        ex = {**ex, "text_ids": None}
    return ex


def validate_examples(examples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [validate_example(ex) for ex in examples]


def make_example(
    example_id: str,
    task: str,
    lang: str,
    texts: list[str],
    roles: list[str],
    teacher_prompt_names: list[str | None],
    student_prompt_names: list[str | None],
    label: float | int | None = None,
    text_ids: list[str] | None = None,
) -> dict[str, Any]:
    return validate_example(
        {
            "example_id": example_id,
            "task": task,
            "lang": lang,
            "texts": texts,
            "roles": roles,
            "teacher_prompt_names": teacher_prompt_names,
            "student_prompt_names": student_prompt_names,
            "label": label,
            "text_ids": text_ids,
        }
    )
