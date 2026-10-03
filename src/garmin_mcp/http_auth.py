"""Protect the HTTP transport with a secret path segment: /<SECRET>/mcp"""
import hmac

from starlette.responses import PlainTextResponse


class PathSecretMiddleware:
    def __init__(self, app, secret: str):
        self.app = app
        self.secret = secret.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/healthz":
            await self.app(scope, receive, send)
            return

        parts = scope["path"].split("/", 2)  # ['', secret, rest]
        candidate = parts[1].encode() if len(parts) > 1 else b""
        if not hmac.compare_digest(candidate, self.secret):
            await PlainTextResponse("Not found", status_code=404)(scope, receive, send)
            return

        new_path = "/" + (parts[2] if len(parts) > 2 else "")
        scope = dict(scope, path=new_path, raw_path=new_path.encode())
        await self.app(scope, receive, send)


def run_with_path_secret(fastmcp, secret: str, host: str, port: int) -> None:
    import uvicorn

    asgi = fastmcp.streamable_http_app()
    asgi.add_middleware(PathSecretMiddleware, secret=secret)
    uvicorn.run(asgi, host=host, port=port, log_level="info")
