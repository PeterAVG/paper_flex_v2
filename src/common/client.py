from typing import Any, Dict, Optional

from requests import Response, Session
from requests.models import HTTPError


class RequestError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class ServerSideError(Exception):
    pass


class BaseAPIClient:
    _session: Optional[Session] = None

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token

    @classmethod
    def session(cls) -> Session:
        if cls._session is None:
            cls._session = Session()
        return cls._session

    def _headers(self) -> Dict[str, str]:
        # return {"Authorization": self.token}
        return {"X-Gravitee-Api-Key": self.token, "Accept": "application/json"}

    def _get(self, endpoint: str, params: Any) -> Response:
        url = f"{self.url}/{endpoint}"
        resp = self.session().get(f"{url}", params=params, headers=self._headers())
        self.handle_errors(resp, url)
        return resp

    def _post(self, endpoint: str, request: Any) -> Response:
        url = f"{self.url}/{endpoint}"
        resp = self.session().post(f"{url}", json=request, headers=self._headers())
        self.handle_errors(resp, url)
        return resp

    @classmethod
    def handle_errors(cls, response: Response, url: str) -> None:
        try:
            response.raise_for_status()
        except HTTPError as e:
            code = response.status_code
            if code == 422 or code == 419 or code == 404:
                raise RequestError(f"Could not handle data: {e}")
            if code == 403 or code == 401:
                raise AuthenticationError(f"Authenticated/Authorization failure, {e}")
            if code >= 500:
                raise ServerSideError(f"{url} gave server failure: {e}")
            raise IOError(f"Failed to retrieve data from {url}: {e}")
