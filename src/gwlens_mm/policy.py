"""Fail-closed deployable-input policy and field-role definitions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Tuple


class FieldRole(str, Enum):
    MODEL_INPUT = "model_input"
    TRAINING_TARGET = "training_target"
    GROUPING_PROVENANCE = "grouping_provenance"
    PRIVILEGED_DIAGNOSTIC = "privileged_diagnostic"


class InputPolicyError(ValueError):
    """Raised when a model-input configuration is forbidden or unknown."""


def _normalized_tokens(name: str) -> Tuple[str, ...]:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()
    return tuple(token for token in re.split(r"[^a-z0-9]+", snake) if token)


@dataclass(frozen=True)
class InputPolicy:
    allowlist: frozenset[str]
    denylist: frozenset[str]
    suspicious_alias_tokens: frozenset[str]
    version: str

    @classmethod
    def from_files(cls, allowlist_path: Path, denylist_path: Path) -> "InputPolicy":
        allow = json.loads(allowlist_path.read_text(encoding="utf-8"))
        deny = json.loads(denylist_path.read_text(encoding="utf-8"))
        if allow["policy_version"] != deny["policy_version"]:
            raise InputPolicyError("allowlist and denylist versions differ")
        allow_fields = frozenset(str(field) for field in allow["fields"])
        deny_fields = frozenset(str(field) for field in deny["exact_fields"])
        overlap = allow_fields & deny_fields
        if overlap:
            raise InputPolicyError(f"fields occur in both policies: {sorted(overlap)}")
        return cls(
            allowlist=allow_fields,
            denylist=deny_fields,
            suspicious_alias_tokens=frozenset(deny["suspicious_alias_tokens"]),
            version=str(allow["policy_version"]),
        )

    def validate_model_inputs(self, fields: Iterable[str]) -> Tuple[str, ...]:
        field_tuple = tuple(fields)
        if len(field_tuple) != len(set(field_tuple)):
            raise InputPolicyError("model-input fields contain duplicates")
        violations = []
        unknown = []
        for field in field_tuple:
            if field in self.denylist:
                violations.append(field)
                continue
            normalized = "_".join(_normalized_tokens(field))
            alias_match = [
                alias
                for alias in self.suspicious_alias_tokens
                if alias in normalized or alias.replace("_", "") in normalized.replace("_", "")
            ]
            if alias_match:
                violations.append(f"{field} (suspicious alias: {sorted(alias_match)[0]})")
            elif field not in self.allowlist:
                unknown.append(field)
        if violations:
            raise InputPolicyError(f"forbidden model inputs: {', '.join(violations)}")
        if unknown:
            raise InputPolicyError(f"unknown, non-allowlisted model inputs: {', '.join(unknown)}")
        return field_tuple

    def validate_roles(self, roles: Mapping[str, FieldRole]) -> None:
        self.validate_model_inputs(
            name for name, role in roles.items() if role is FieldRole.MODEL_INPUT
        )
