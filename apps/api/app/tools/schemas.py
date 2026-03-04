from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


ToolImplementationKind = Literal["builtin", "python_module", "http", "prompt"]
ToolKind = Literal["local", "external"]
ToolType = Literal["builtin", "http", "python", "prompt"]


class ToolSpecModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tool_id: str = ""
    id: str = ""
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    implementation_kind: ToolImplementationKind
    implementation_ref: str
    enabled: bool = True
    version: int = 1
    deleted_at: Optional[str] = None
    source: str = "custom"
    created_at: str = ""
    updated_at: str = ""
    deprecated_reason: Optional[str] = None
    kind: str = "external"
    type: str = "http"
    config: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_ids(self) -> "ToolSpecModel":
        normalized = str(self.tool_id or self.id or "").strip()
        self.tool_id = normalized
        self.id = normalized
        self.name = str(self.name or normalized)
        self.description = str(self.description or "")
        self.implementation_ref = str(self.implementation_ref or "")
        self.kind = str(self.kind or "external")
        self.type = str(self.type or "http")
        return self


class ToolListResponseModel(BaseModel):
    tools: list[ToolSpecModel] = Field(default_factory=list)


class ToolInvokeRequestModel(BaseModel):
    tool_id: str
    input: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None


class ToolInvokeResponseModel(BaseModel):
    run_id: str
    tool_id: str
    status: str
    output: Any = None
    steps: list[Dict[str, Any]] = Field(default_factory=list)


class CreateToolRequestModel(BaseModel):
    name: str
    id: Optional[str] = None
    kind: ToolKind = "external"
    description: str = ""
    type: Literal["http", "python", "prompt"]
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)


class UpdateToolRequestModel(BaseModel):
    enabled: bool
