"""Deterministic role-specific evidence assembly."""
from ..types import Role
from .evidence import Evidence
from .geometry import kilometers


def bind(tools, request, task, role, operation, shared):
    if operation not in tools.operations(request, task, role):
        raise ValueError("operation violates role or content permissions")
    evidence = Evidence()
    if operation == "refine":
        for p in task.targets:
            evidence.candidate_text[p] = ["Reassess the admitted task evidence for this target."]
        return evidence
    for p in task.targets:
        poi = request.candidates[p]
        lines = []
        if role in {Role.PREFERENCE, Role.TRAJECTORY}:
            history = list(request.history)
            if operation == "relevant_history":
                history = [v for v in history if v.poi.category == poi.category] or history
            if role == Role.TRAJECTORY:
                history = history[-10:]
            if operation == "retrieve_sequence":
                tools.retrieve(request, role, p, shared, evidence, operation)
                continue
            lines = [f"candidate={poi.poi_id}; category={poi.category}"]
            for v in history:
                # Preference may read available text; trajectory only sees order/categories/gaps.
                text = f"; text={v.poi.text}" if role == Role.PREFERENCE and v.poi.mask[0] else ""
                lines.append(f"visit={v.poi.poi_id}; category={v.poi.category}; time={v.timestamp}{text}")
                if role == Role.PREFERENCE and v.poi.mask[1]:
                    evidence.candidate_images.setdefault(p, []).append(v.poi.image)
            if role == Role.PREFERENCE and poi.mask[0]:
                lines.append(f"candidate text={poi.text}")
        elif role == Role.SPATIOTEMPORAL:
            lines = [f"candidate={poi.poi_id}; known_time={request.cutoff}; distance_km={kilometers(request.location, poi):.4f}"]
            if operation == "transitions":
                lines.extend(f"past={v.poi.poi_id}; category={v.poi.category}; time={v.timestamp}; lat={v.poi.latitude}; lon={v.poi.longitude}"
                             for v in request.history)
        elif role == Role.TEXT:
            if poi.mask[0]:
                lines = [f"candidate={poi.poi_id}; text={poi.text}"]
        elif role == Role.VISUAL:
            images = []
            if operation != "history_images" and poi.mask[1]:
                images.append(poi.image)
                evidence.objects.append(f"image:{poi.poi_id}")
            if operation != "target_images":
                for v in request.history:
                    if v.poi.mask[1]:
                        images.append(v.poi.image)
                        evidence.objects.append(f"image:{v.poi.poi_id}")
            if images:
                evidence.candidate_images[p] = images
                lines = [f"candidate={poi.poi_id}; target_image_available={poi.mask[1]}; history_images={len(images)}"]
        elif role == Role.COLLABORATIVE:
            tools.retrieve(request, role, p, shared, evidence, operation)
            continue
        if lines:
            evidence.candidate_text[p] = lines
            evidence.objects.append(f"poi:{poi.poi_id}")
    if role in {Role.PREFERENCE, Role.TRAJECTORY, Role.SPATIOTEMPORAL}:
        evidence.objects.extend(f"visit:{v.poi.poi_id}:{v.timestamp}" for v in request.history)
    if not evidence.coverage:
        evidence.status = "empty"
    return evidence
