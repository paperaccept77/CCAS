"""Evidence tool registry and permission-aware facade."""
from .index import TrainingIndex
from . import permissions
from .binding import bind


class EvidenceTools:
    def __init__(self, config, index_path=None, device="cpu"):
        self.config = config
        self.retrieval = TrainingIndex(config, index_path, device)
        self.index = self.retrieval.index

    def eligible_records(self, request, role):
        return self.retrieval.eligible_records(request, role)

    def operations(self, request, task, role):
        return permissions.operations(self, request, task, role)

    def eligible(self, request, task, role):
        return permissions.eligible(self, request, task, role)

    def bind(self, request, task, role, operation, shared):
        return bind(self, request, task, role, operation, shared)

    def retrieve(self, request, role, p, shared, evidence, operation):
        return self.retrieval.retrieve(request, role, p, shared, evidence, operation)
