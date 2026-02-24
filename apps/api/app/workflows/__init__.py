from app.workflows.schemas import (
    ToolCallApprovalRequest,
    ValidationDiagnosticModel,
    WorkflowCompileRequestV2,
    WorkflowCompileResponseV2,
    WorkflowRunRequestV2,
    WorkflowRunResponseV2,
)


def compile_workflow_spec(*args, **kwargs):
    from app.workflows.service import compile_workflow_spec as _compile_workflow_spec

    return _compile_workflow_spec(*args, **kwargs)


def create_workflow_version(*args, **kwargs):
    from app.workflows.service import create_workflow_version as _create_workflow_version

    return _create_workflow_version(*args, **kwargs)


def get_latest_workflow_payload(*args, **kwargs):
    from app.workflows.service import get_latest_workflow_payload as _get_latest_workflow_payload

    return _get_latest_workflow_payload(*args, **kwargs)


def get_workflow_versions_payload(*args, **kwargs):
    from app.workflows.service import (
        get_workflow_versions_payload as _get_workflow_versions_payload,
    )

    return _get_workflow_versions_payload(*args, **kwargs)

__all__ = [
    "ToolCallApprovalRequest",
    "ValidationDiagnosticModel",
    "WorkflowCompileRequestV2",
    "WorkflowCompileResponseV2",
    "WorkflowRunRequestV2",
    "WorkflowRunResponseV2",
    "compile_workflow_spec",
    "create_workflow_version",
    "get_latest_workflow_payload",
    "get_workflow_versions_payload",
]
