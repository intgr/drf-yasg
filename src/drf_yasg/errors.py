from typing import Any, Optional
class SwaggerError(Exception):
    pass


class SwaggerValidationError(SwaggerError):
    errors: Any
    spec: Any
    source_codec: Any
    def __init__(self, msg: Any, errors: Optional[Any] = None, spec: Optional[Any] = None, source_codec: Optional[Any] = None, *args: Any) -> None:
        super(SwaggerValidationError, self).__init__(msg, *args)
        self.errors = errors
        self.spec = spec
        self.source_codec = source_codec


class SwaggerGenerationError(SwaggerError):
    pass
