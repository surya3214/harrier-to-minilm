"""Teacher (Harrier-native) and student (minimized) prompt maps."""

from __future__ import annotations

from typing import Any


DEFAULT_TEACHER_PROMPTS = {
    "sts_query": "Instruct: Retrieve semantically similar text\nQuery: ",
    "web_search_query": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    ),
    "bitext_query": "Instruct: Retrieve parallel sentences\nQuery: ",
    "document": "",
}

DEFAULT_STUDENT_PROMPTS = {
    "sts": "sts: ",
    "ret": "ret: ",
    "bitext": "bitext: ",
    "document": "",
}


class PromptMap:
    def __init__(self, teacher: dict[str, str], student: dict[str, str]):
        self.teacher = {**DEFAULT_TEACHER_PROMPTS, **(teacher or {})}
        self.student = {**DEFAULT_STUDENT_PROMPTS, **(student or {})}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PromptMap":
        prompts = cfg.get("prompts", {})
        return cls(prompts.get("teacher", {}), prompts.get("student", {}))

    def teacher_text(self, prompt_name: str | None, text: str) -> str:
        if not prompt_name:
            return text
        prefix = self.teacher.get(prompt_name, "")
        return f"{prefix}{text}" if prefix else text

    def student_text(self, prompt_name: str | None, text: str) -> str:
        if not prompt_name:
            return text
        prefix = self.student.get(prompt_name, "")
        return f"{prefix}{text}" if prefix else text
