"""Role-adapted continuous candidate and summary readouts."""
import torch
from torch import nn
from ..types import OperationOutput
from .backbone import load_backbone


class LLMExecutor(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.tokenizer, self.backbone, hidden = load_backbone(config)
        self.image_projection = nn.Linear(config.image_dim, hidden)
        self.evidence_projection = nn.Linear(config.dim, hidden)
        self.state_projection = nn.Linear(3 * config.dim, hidden)
        self.message_readout = nn.Linear(hidden, config.dim)
        self.summary_readout = nn.Linear(hidden, config.dim)
        self.candidate_marker = nn.Parameter(torch.zeros(hidden))
        self.summary_marker = nn.Parameter(torch.zeros(hidden))
        self.adapters_trainable = True
        self.to(device)

    def set_execution_trainable(self, enabled):
        self.adapters_trainable = enabled
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(enabled and (not name.startswith("backbone.") or "lora_" in name))

    def forward(self, role, operation, task, request, evidence, memory, local, shared):
        d = self.config.dim
        if not evidence.coverage:
            return OperationOutput({}, memory.new_zeros(d), "empty", evidence.objects)
        self.backbone.set_adapter(role.name.lower())
        # set_adapter can change trainability; enforce the frozen-PPO boundary every call.
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(self.adapters_trainable and "lora_" in name)
        embed = self.backbone.get_input_embeddings()
        dtype, device = embed.weight.dtype, embed.weight.device
        prompt = (f"Role: {role.name}. Operation: {operation}. Task: {task.kind.name}. "
                  "Analyze available pre-prediction evidence for next-POI ranking. "
                  "Missing content stays missing. Candidate readouts must represent evidence.\n")
        covered = evidence.coverage
        prefix = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # Reserve context and all readout markers before admitting evidence chunks.
        remaining = self.config.input_tokens - len(prefix) - 1 - 2 * len(covered) - 1
        if remaining <= 0:
            raise ValueError("candidate readouts exceed the token budget; lower pool size")
        chunks = [embed(torch.tensor(prefix, device=device)),
                  self.state_projection(torch.cat((memory, local, shared))).to(dtype).unsqueeze(0)]
        # Bound each candidate equally so early candidates do not consume every token.
        quota = remaining // len(covered)
        admitted = []
        for p in covered:
            vectors = [self.image_projection(v) for v in evidence.candidate_images.get(p, [])]
            vectors += [self.evidence_projection(v) for v in evidence.retrieval_vectors.get(p, [])]
            text = f"Target {p} ({request.candidates[p].poi_id}):\n" + "\n".join(evidence.candidate_text.get(p, []))
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            n_vectors = min(len(vectors), quota // 2)
            text_tokens = tokens[:max(0, quota - n_vectors)]
            if not text_tokens and n_vectors == 0:
                continue
            admitted.append(p)
            if text_tokens:
                chunks.append(embed(torch.tensor(text_tokens, device=device)))
            if n_vectors:
                chunks.append(torch.stack(vectors[:n_vectors]).to(dtype))
        if not admitted:
            return OperationOutput({}, memory.new_zeros(d), "empty", evidence.objects)
        positions = []
        length = sum(x.shape[0] for x in chunks)
        for p in admitted:
            # Explicit candidate identity precedes each readout, outside truncated evidence.
            marker_id = self.tokenizer(str(p), add_special_tokens=False)["input_ids"][:1]
            chunks.append(embed(torch.tensor(marker_id, device=device)))
            length += 1
            positions.append(length)
            chunks.append(self.candidate_marker.to(dtype).unsqueeze(0))
            length += 1
        summary_position = length
        chunks.append(self.summary_marker.to(dtype).unsqueeze(0))
        inputs = torch.cat(chunks).unsqueeze(0)
        outputs = self.backbone(inputs_embeds=inputs, attention_mask=torch.ones(inputs.shape[:2], device=device),
                                use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state[0].float()
        messages = {p: self.message_readout(hidden[position]) for p, position in zip(admitted, positions)}
        summary = self.summary_readout(hidden[summary_position])
        return OperationOutput(messages, summary, "ok", sorted(set(evidence.objects)), input_tokens=inputs.shape[1])

    def compact_state_dict(self):
        return {key: value for key, value in self.state_dict().items()
                if not key.startswith("backbone.") or "lora_" in key}
