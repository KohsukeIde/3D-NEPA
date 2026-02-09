from typing import Any, Dict, Protocol


class QueryBackend(Protocol):
    def get_pools(self) -> Dict[str, Any]:
        raise NotImplementedError
