"""CATH engine facade."""
from . import tasks, observations, episode, rounds


class Engine:
    def __init__(self, config, model, policy, executor, tools, cost_profile):
        self.config, self.model, self.policy = config, model, policy
        self.executor, self.tools, self.cost = executor, tools, cost_profile

    def eligible_roles(self, request, task):
        return tasks.eligible_roles(self, request, task)

    def cheapest_operation(self, request, task, members):
        return tasks.cheapest_operation(self, request, task, members)

    def execution_reserve(self, request, task, members):
        return tasks.execution_reserve(self, request, task, members)

    def legal_tasks(self, request, state):
        return tasks.legal_tasks(self, request, state)

    def action(self, state, first=None, second=None, third=None, numeric=(0, 0, 0, 0)):
        return observations.action(self, state, first, second, third, numeric)

    def observation(self, request, state, ledger, kind, actions, keys, mask, members=(), role=None):
        return observations.observation(self, request, state, ledger, kind, actions, keys, mask, members, role)

    def task_observation(self, request, state, ledger, tasks):
        return observations.task_observation(self, request, state, ledger, tasks)

    def member_observation(self, request, state, ledger, task, members, initiator):
        return observations.member_observation(self, request, state, ledger, task, members, initiator)

    def operation_observation(self, request, state, ledger, task, role, local, summary, round_budget, round_number, members):
        return observations.operation_observation(self, request, state, ledger, task, role, local, summary, round_budget, round_number, members)

    def run(self, request, mode="greedy", script=None):
        return episode.run(self, request, mode, script)

    def execute_task(self, request, state, ledger, task, members, choose):
        return rounds.execute_task(self, request, state, ledger, task, members, choose)
