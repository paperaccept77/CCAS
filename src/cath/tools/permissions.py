"""Role and modality operation masks."""
from ..types import OPERATIONS, Role


def operations(tools, request, task, role):
    targets = [request.candidates[i] for i in task.targets]
    history = request.history
    available = []
    for op in OPERATIONS[role]:
        legal = True
        if op in {"refine", "request_peer", "finish"}:
            available.append(op)
            continue
        if role == Role.PREFERENCE:
            legal = bool(history)
        elif role == Role.TRAJECTORY:
            legal = bool(tools.eligible_records(request, role)) if op == "retrieve_sequence" else bool(history)
        elif role == Role.SPATIOTEMPORAL:
            legal = bool(history) if op == "transitions" else True
        elif role == Role.TEXT:
            legal = any(p.mask[0] for p in targets)
        elif role == Role.VISUAL:
            target_images = any(p.mask[1] for p in targets)
            history_images = any(v.poi.mask[1] for v in history)
            legal = target_images if op == "target_images" else history_images if op == "history_images" else target_images or history_images
        elif role == Role.COLLABORATIVE:
            legal = bool(tools.eligible_records(request, role))
        if legal:
            available.append(op)
    return available


def eligible(tools, request, task, role):
    return any(op not in {"refine", "request_peer", "finish"} for op in tools.operations(request, task, role))
