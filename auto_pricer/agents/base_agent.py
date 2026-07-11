class BaseAgent:
    """Consistent interface for all AutoPricer agents. Every agent takes a
    plain-text car description and returns a dollar price estimate."""

    def price(self, description: str) -> float:
        raise NotImplementedError
