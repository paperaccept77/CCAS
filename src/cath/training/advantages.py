"""Undiscounted generalized advantage estimation."""



def gae(decisions, gae_lambda=0.95):
    """Undiscounted GAE, gamma=1; episodes must end in a terminal decision."""
    advantage, next_value, values = 0.0, 0.0, []
    for decision in reversed(decisions):
        continuation = 0.0 if decision.done else 1.0
        delta = decision.reward + continuation * next_value - decision.value
        advantage = delta + continuation * gae_lambda * advantage
        values.append((advantage, advantage + decision.value))
        next_value = decision.value
    return list(reversed(values))
