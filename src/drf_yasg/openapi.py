import collections
import logging
import re
import urllib.parse as urlparse
from collections import OrderedDict
from typing import overload, Any

from django.urls import get_script_prefix
from django.utils.functional import Promise
from inflection import camelize

from .utils import dict_has_ordered_keys, filter_none, force_real_str

from types import ModuleType
from typing import Any, Optional, Union, List, Dict, Tuple, Type, ClassVar, Callable, KeysView, Iterable, Iterator
from rest_framework.serializers import Serializer
from typing_extensions import Literal
FORMAT_DATE: Literal['date']
FORMAT_EMAIL: Literal['email']
FORMAT_UUID: Literal['uuid']
try:
    from collections import abc as collections_abc
except ImportError:
    collections_abc: ModuleType = collections

logger = logging.getLogger(__name__)

TYPE_OBJECT: Literal['object'] = "object"  #:
TYPE_STRING: Literal['string'] = "string"  #:
TYPE_NUMBER: Literal['number'] = "number"  #:
TYPE_INTEGER: Literal['integer'] = "integer"  #:
TYPE_BOOLEAN: Literal['boolean'] = "boolean"  #:
TYPE_ARRAY: Literal['array'] = "array"  #:
TYPE_FILE: Literal['file'] = "file"  #:

# officially supported by Swagger 2.0 spec
FORMAT_DATE = "date"  #:
FORMAT_DATETIME: Literal['date-time'] = "date-time"  #:
FORMAT_PASSWORD: Literal['password'] = "password"  #:
FORMAT_BINARY: Literal['binary'] = "binary"  #:
FORMAT_BASE64: Literal['bytes'] = "bytes"  #:
FORMAT_FLOAT: Literal['float'] = "float"  #:
FORMAT_DOUBLE: Literal['double'] = "double"  #:
FORMAT_INT32: Literal['int32'] = "int32"  #:
FORMAT_INT64: Literal['int64'] = "int64"  #:

# defined in JSON-schema
FORMAT_EMAIL = "email"  #:
FORMAT_IPV4: Literal['ipv4'] = "ipv4"  #:
FORMAT_IPV6: Literal['ipv6'] = "ipv6"  #:
FORMAT_URI: Literal['uri'] = "uri"  #:

# pulled out of my ass
FORMAT_UUID = "uuid"  #:
FORMAT_SLUG: Literal['slug'] = "slug"  #:
FORMAT_DECIMAL: Literal['decimal'] = "decimal"

IN_BODY: Literal['body'] = 'body'  #:
IN_PATH: Literal['path'] = 'path'  #:
IN_QUERY: Literal['query'] = 'query'  #:
IN_FORM: Literal['formData'] = 'formData'  #:
IN_HEADER: Literal['header'] = 'header'  #:

SCHEMA_DEFINITIONS: Literal['definitions'] = 'definitions'  #:


def make_swagger_name(attribute_name: str) -> str:
    """
    Convert a python variable name into a Swagger spec attribute name.

    In particular,
     * if name starts with ``x_``, return ``x-{camelCase}``
     * if name is ``ref``, return ``$ref``
     * else return the name converted to camelCase, with trailing underscores stripped

    :param str attribute_name: python attribute name
    :return: swagger name
    """
    if attribute_name == 'ref':
        return "$ref"
    if attribute_name.startswith("x_"):
        return "x-" + camelize(attribute_name[2:], uppercase_first_letter=False)
    return camelize(attribute_name.rstrip('_'), uppercase_first_letter=False)


def _bare_SwaggerDict(cls):
    assert issubclass(cls, SwaggerDict)
    result = cls.__new__(cls)
    OrderedDict.__init__(result)  # no __init__ called for SwaggerDict subclasses!
    return result


class SwaggerDict(OrderedDict):
    """A particular type of OrderedDict, which maps all attribute accesses to dict lookups using
     :func:`.make_swagger_name`. Attribute names starting with ``_`` are set on the object as-is and are not included
     in the specification output.

     Used as a base class for all Swagger helper models.
    """
    _extras__: Dict[str, Any]
    def __init__(self, **attrs: Any) -> None:
        super(SwaggerDict, self).__init__()
        self._extras__ = attrs
        if type(self) == SwaggerDict:
            self._insert_extras__()

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super(SwaggerDict, self).__setattr__(key, value)
            return
        if value is not None:
            self[make_swagger_name(key)] = value

    def __getattr__(self, item: str) -> Any:
        if item.startswith('_'):
            raise AttributeError
        try:
            return self[make_swagger_name(item)]
        except KeyError:
            # raise_from is EXTREMELY slow, replaced with plain raise
            raise AttributeError("object of class " + type(self).__name__ + " has no attribute " + item)

    def __delattr__(self, item):
        if item.startswith('_'):
            super(SwaggerDict, self).__delattr__(item)
            return
        del self[make_swagger_name(item)]

    def _insert_extras__(self) -> None:
        """
        From an ordering perspective, it is desired that extra attributes such as vendor extensions stay at the
        bottom of the object. However, python2.7's OrderdDict craps out if you try to insert into it before calling
        init. This means that subclasses must call super().__init__ as the first statement of their own __init__,
        which would result in the extra attributes being added first. For this reason, we defer the insertion of the
        attributes and require that subclasses call ._insert_extras__ at the end of their __init__ method.
        """
        for attr, val in sorted(self._extras__.items()):
            setattr(self, attr, val)

    @staticmethod
    def _as_odict(obj, memo):
        """Implementation detail of :meth:`.as_odict`"""
        if id(obj) in memo:
            return memo[id(obj)]

        if isinstance(obj, Promise) and hasattr(obj, '_proxy____cast'):
            # handle __proxy__ objects from django.utils.functional.lazy
            obj = obj._proxy____cast()

        if isinstance(obj, collections_abc.Mapping):
            result = OrderedDict()
            memo[id(obj)] = result
            items = obj.items()
            if not dict_has_ordered_keys(obj):
                items = sorted(items)
            for attr, val in items:
                result[attr] = SwaggerDict._as_odict(val, memo)
            return result
        elif isinstance(obj, str):
            return force_real_str(obj)
        elif isinstance(obj, collections_abc.Iterable) and not isinstance(obj, collections_abc.Iterator):
            return type(obj)(SwaggerDict._as_odict(elem, memo) for elem in obj)

        return obj

    def as_odict(self) -> OrderedDict:
        """Convert this object into an ``OrderedDict`` instance.

        :rtype: OrderedDict
        """
        return SwaggerDict._as_odict(self, {})

    def __reduce__(self):
        # for pickle supprt; this skips calls to all SwaggerDict __init__ methods and relies
        # on the already set attributes instead
        attrs = {k: v for k, v in vars(self).items() if not k.startswith('_NP_')}
        return _bare_SwaggerDict, (type(self),), attrs, None, iter(self.items())


class Contact(SwaggerDict):
    name: Optional[str]
    url: Optional[str]
    email: Optional[str]
    def __init__(self, name: Optional[str] = None, url: Optional[str] = None, email: Optional[str] = None, **extra: Any) -> None:
        """Swagger Contact object

        At least one of the following fields is required:

        :param str name: contact name
        :param str url: contact url
        :param str email: contact e-mail
        """
        super(Contact, self).__init__(**extra)
        if name is None and url is None and email is None:
            raise AssertionError("one of name, url or email is requires for Swagger Contact object")
        self.name = name
        self.url = url
        self.email = email
        self._insert_extras__()


class License(SwaggerDict):
    name: str
    url: Optional[str]
    def __init__(self, name: str, url: Optional[str] = None, **extra: Any) -> None:
        """Swagger License object

        :param str name: Required. License name
        :param str url: link to detailed license information
        """
        super(License, self).__init__(**extra)
        if name is None:
            raise AssertionError("name is required for Swagger License object")
        self.name = name
        self.url = url
        self._insert_extras__()


class Info(SwaggerDict):
    title: str
    _default_version: str
    description: Optional[str]
    terms_of_service: Optional[str]
    contact: Optional[Contact]
    license: Optional[License]
    def __init__(self, title: str, default_version: str, description: Optional[str] = None, terms_of_service: Optional[str] = None, contact: Optional[Contact] = None, license: Optional[License] = None,
                 **extra: Any) -> None:
        """Swagger Info object

        :param str title: Required. API title.
        :param str default_version: Required. API version string (not to be confused with Swagger spec version)
        :param str description: API description; markdown supported
        :param str terms_of_service: API terms of service; should be a URL
        :param Contact contact: contact object
        :param License license: license object
        """
        super(Info, self).__init__(**extra)
        if title is None or default_version is None:
            raise AssertionError("title and version are required for Swagger info object")
        if contact is not None and not isinstance(contact, Contact):
            raise AssertionError("contact must be a Contact object")
        if license is not None and not isinstance(license, License):
            raise AssertionError("license must be a License object")
        self.title = title
        self._default_version = default_version
        self.description = description
        self.terms_of_service = terms_of_service
        self.contact = contact
        self.license = license
        self._insert_extras__()


class Swagger(SwaggerDict):
    swagger: str
    info: Optional[Info]
    base_path: Optional[str]
    consumes: Optional[List[str]]
    produces: Optional[List[str]]
    security_definitions: Optional[Dict[str, dict]]
    security: Optional[Dict[str, List[str]]]
    paths: Optional[Paths]
    definitions: Optional[Dict[str, Schema]]
    host: str
    schemes: List[str]
    def __init__(self, info: Optional[Info] = None, _url: Optional[str] = None, _prefix: Optional[str] = None, _version: Optional[str] = None, consumes: Optional[List[str]] = None, produces: Optional[List[str]] = None,
                 security_definitions: Optional[Dict[str, dict]] = None, security: Optional[Dict[str, List[str]]] = None, paths: Optional[Paths] = None, definitions: Optional[Dict[str, Schema]] = None, **extra: Any) -> None:
        """Root Swagger object.

        :param .Info info: info object
        :param str _url: URL used for setting the API host and scheme
        :param str _prefix: api path prefix to use in setting basePath; this will be appended to the wsgi
            SCRIPT_NAME prefix or Django's FORCE_SCRIPT_NAME if applicable
        :param str _version: version string to override Info
        :param dict[str,dict] security_definitions: list of supported authentication mechanisms
        :param list[dict[str,list[str]]] security: authentication mechanisms accepted globally
        :param list[str] consumes: consumed MIME types; can be overriden in Operation
        :param list[str] produces: produced MIME types; can be overriden in Operation
        :param Paths paths: paths object
        :param dict[str,Schema] definitions: named models
        """
        super(Swagger, self).__init__(**extra)
        self.swagger = '2.0'
        self.info = info
        self.info.version = _version or info._default_version

        if _url:
            url = urlparse.urlparse(_url)
            assert url.netloc and url.scheme, "if given, url must have both schema and netloc"
            self.host = url.netloc
            self.schemes = [url.scheme]

        self.base_path = self.get_base_path(get_script_prefix(), _prefix)
        self.consumes = consumes
        self.produces = produces
        self.security_definitions = filter_none(security_definitions)
        self.security = filter_none(security)
        self.paths = paths
        self.definitions = filter_none(definitions)
        self._insert_extras__()

    @classmethod
    def get_base_path(cls, script_prefix: str, api_prefix: str) -> str:
        """Determine an appropriate value for ``basePath`` based on the SCRIPT_NAME and the api common prefix.

        :param str script_prefix: script prefix as defined by django ``get_script_prefix``
        :param str api_prefix: api common prefix
        :return: joined base path
        """
        # avoid double slash when joining script_name with api_prefix
        if script_prefix and script_prefix.endswith('/'):
            script_prefix = script_prefix[:-1]
        if not api_prefix.startswith('/'):
            api_prefix = '/' + api_prefix

        base_path = script_prefix + api_prefix

        # ensure that the base path has a leading slash and no trailing slash
        if base_path and base_path.endswith('/'):
            base_path = base_path[:-1]
        if not base_path.startswith('/'):
            base_path = '/' + base_path

        return base_path


class Paths(SwaggerDict):
    def __init__(self, paths: Dict[str, PathItem], **extra: Any) -> None:
        """A listing of all the paths in the API.

        :param dict[str,PathItem] paths:
        """
        super(Paths, self).__init__(**extra)
        for path, path_obj in paths.items():
            assert path.startswith("/")
            if path_obj is not None:  # pragma: no cover
                self[path] = path_obj
        self._insert_extras__()


class PathItem(SwaggerDict):
    get: Optional[Operation]
    head: Optional[Operation]
    post: Optional[Operation]
    put: Optional[Operation]
    patch: Optional[Operation]
    delete: Optional[Operation]
    options: Optional[Operation]
    parameters: Optional[List[Parameter]]
    OPERATION_NAMES: ClassVar[List[str]] = ['get', 'put', 'post', 'delete', 'options', 'head', 'patch']

    def __init__(self, get: Optional[Operation] = None, put: Optional[Operation] = None, post: Optional[Operation] = None, delete: Optional[Operation] = None, options: Optional[Operation] = None,
                 head: Optional[Operation] = None, patch: Optional[Operation] = None, parameters: Optional[List[Parameter]] = None, **extra: Any) -> None:
        """Information about a single path

        :param Operation get: operation for GET
        :param Operation put: operation for PUT
        :param Operation post: operation for POST
        :param Operation delete: operation for DELETE
        :param Operation options: operation for OPTIONS
        :param Operation head: operation for HEAD
        :param Operation patch: operation for PATCH
        :param list[Parameter] parameters: parameters that apply to all operations
        """
        super(PathItem, self).__init__(**extra)
        self.get = get
        self.head = head
        self.post = post
        self.put = put
        self.patch = patch
        self.delete = delete
        self.options = options
        self.parameters = filter_none(parameters)
        self._insert_extras__()

    @property
    def operations(self) -> List[Tuple[str, Operation]]:
        """A list of all standard Operations on this PathItem object. See :attr:`.OPERATION_NAMES`.

        :return: list of (method name, Operation) tuples
        :rtype: list[tuple[str,Operation]]
        """
        return [(k, v) for k, v in self.items() if k in PathItem.OPERATION_NAMES and v]


class Operation(SwaggerDict):
    operation_id: str
    responses: Responses
    parameters: Optional[List[Parameter]]
    consumes: Optional[List[str]]
    produces: Optional[List[str]]
    summary: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    security: Optional[Dict[str, List[str]]]
    def __init__(self, operation_id: str, responses: Responses, parameters: Optional[List[Parameter]] = None, consumes: Optional[List[str]] = None, produces: Optional[List[str]] = None, summary: Optional[str] = None,
                 description: Optional[str] = None, tags: Optional[List[str]] = None, security: Optional[Dict[str, List[str]]] = None, **extra: Any) -> None:
        """Information about an API operation (path + http method combination)

        :param str operation_id: operation ID, should be unique across all operations
        :param Responses responses: responses returned
        :param list[Parameter] parameters: parameters accepted
        :param list[str] consumes: content types accepted
        :param list[str] produces: content types produced
        :param str summary: operation summary; should be < 120 characters
        :param str description: operation description; can be of any length and supports markdown
        :param list[str] tags: operation tags
        :param list[dict[str,list[str]]] security: list of security requirements
        """
        super(Operation, self).__init__(**extra)
        self.operation_id = operation_id
        self.summary = summary
        self.description = description
        self.parameters = filter_none(parameters)
        self.responses = responses
        self.consumes = filter_none(consumes)
        self.produces = filter_none(produces)
        self.tags = filter_none(tags)
        self.security = filter_none(security)
        self._insert_extras__()


def _check_type(type, format, enum, pattern, items, _obj_type):
    if items and type != TYPE_ARRAY:
        raise AssertionError("items can only be used when type is array")
    if type == TYPE_ARRAY and not items:
        raise AssertionError("TYPE_ARRAY requires the items attribute")
    if pattern and type != TYPE_STRING:
        raise AssertionError("pattern can only be used when type is string")
    if (format or enum or pattern) and type in (TYPE_OBJECT, TYPE_ARRAY, None):
        raise AssertionError("[format, enum, pattern] can only be applied to primitive " + _obj_type)


_EnumType = Union[List[str], List[int], List[bool], List[float]]
_FORMAT_any = Literal['date', 'date-time', 'password', 'binary', 'bytes', 'float', 'double', 'int32', 'int64', 'email', 'ipv4', 'ipv6', 'uri', 'uuid', 'slug', 'decimal']
_TYPE_any = Literal['object', 'string', 'number', 'integer', 'boolean', 'array', 'file']
class Items(SwaggerDict):
    type: _TYPE_any
    format: Optional[_FORMAT_any]
    enum: Optional[_EnumType]
    pattern: Optional[str]
    items_: Optional[Items]
    def __init__(self, type: _TYPE_any = None, format: Optional[_FORMAT_any] = None, enum: Optional[_EnumType] = None, pattern: Optional[str] = None, items: Optional[Items] = None, **extra: Any) -> None:
        """Used when defining an array :class:`.Parameter` to describe the array elements.

        :param str type: type of the array elements; must not be ``object``
        :param str format: value format, see OpenAPI spec
        :param list enum: restrict possible values
        :param str pattern: pattern if type is ``string``
        :param .Items items: only valid if `type` is ``array``
        """
        super(Items, self).__init__(**extra)
        assert type is not None, "type is required!"
        self.type = type
        self.format = format
        self.enum = enum
        self.pattern = pattern
        self.items_ = items
        self._insert_extras__()
        _check_type(type, format, enum, pattern, items, self.__class__)


_SchemaOrRef = Union[Schema, SchemaRef]
_IN_any = Literal['body', 'path', 'query', 'formData', 'header']
class Parameter(SwaggerDict):
    name: str
    in_: _IN_any
    description: Optional[str]
    required: Optional[bool]
    schema: Optional[_SchemaOrRef]
    type: Optional[_TYPE_any]
    format: Optional[_FORMAT_any]
    enum: Optional[_EnumType]
    pattern: Optional[str]
    items_: Optional[Items]
    default: Optional[Any]
    def __init__(self, name: str, in_: _IN_any, description: Optional[str] = None, required: Optional[bool] = None, schema: Optional[_SchemaOrRef] = None,
                 type: Optional[_TYPE_any] = None, format: Optional[_FORMAT_any] = None, enum: Optional[_EnumType] = None, pattern: Optional[str] = None, items: Optional[Items] = None, default: Optional[Any] = None, **extra: Any) -> None:
        """Describe parameters accepted by an :class:`.Operation`. Each parameter should be a unique combination of
        (`name`, `in_`). ``body`` and ``form`` parameters in the same operation are mutually exclusive.

        :param str name: parameter name
        :param str in_: parameter location
        :param str description: parameter description
        :param bool required: whether the parameter is required for the operation
        :param schema: required if `in_` is ``body``
        :type schema: Schema or SchemaRef
        :param str type: parameter type; required if `in_` is not ``body``; must not be ``object``
        :param str format: value format, see OpenAPI spec
        :param list enum: restrict possible values
        :param str pattern: pattern if type is ``string``
        :param .Items items: only valid if `type` is ``array``
        :param default: default value if the parameter is not provided; must conform to parameter type
        """
        super(Parameter, self).__init__(**extra)
        self.name = name
        self.in_ = in_
        self.description = description
        self.required = required
        self.schema = schema
        self.type = type
        self.format = format
        self.enum = enum
        self.pattern = pattern
        self.items_ = items
        self.default = default
        self._insert_extras__()
        if (not schema and not type) or (schema and type):
            raise AssertionError("either schema or type are required for Parameter object (not both)!")
        if schema and isinstance(schema, Schema):
            schema._remove_read_only()
        if self['in'] == IN_PATH:
            # path parameters must always be required
            assert required is not False, "path parameter cannot be optional"
            self.required = True
        if self['in'] != IN_BODY and schema is not None:
            raise AssertionError("schema can only be applied to a body Parameter, not %s" % type)
        if default and not type:
            raise AssertionError("default can only be applied to a non-body Parameter")
        _check_type(type, format, enum, pattern, items, self.__class__)


class Schema(SwaggerDict):
    title: Optional[str]
    description: Optional[str]
    type: Optional[_TYPE_any]
    format: Optional[_FORMAT_any]
    enum: Optional[_EnumType]
    pattern: Optional[str]
    properties: Optional[Any]
    additional_properties: Optional[Any]
    required: Optional[Any]
    items_: Optional[Any]
    default: Optional[Any]
    read_only: Optional[bool]
    OR_REF: ClassVar[Tuple[Type[Schema], Type[SchemaRef]]] = ()  #: useful for type-checking, e.g ``isinstance(obj, openapi.Schema.OR_REF)``

    def __init__(self, title: Optional[str] = None, description: Optional[str] = None, type: Optional[_TYPE_any] = None, format: Optional[_FORMAT_any] = None, enum: Optional[_EnumType] = None, pattern: Optional[str] = None, properties: Optional[Any] = None,
                 additional_properties: Optional[Any] = None, required: Optional[List[str]] = None, items: Optional[Any] = None, default: Optional[Any] = None, read_only: Optional[bool] = None, **extra: Any) -> None:
        """Describes a complex object accepted as parameter or returned as a response.

        :param str title: schema title
        :param str description: schema description
        :param str type: value type; required
        :param str format: value format, see OpenAPI spec
        :param list enum: restrict possible values
        :param str pattern: pattern if type is ``string``
        :param properties: object properties; required if `type` is ``object``
        :type properties: dict[str,Schema or SchemaRef]
        :param additional_properties: allow wildcard properties not listed in `properties`
        :type additional_properties: bool or Schema or SchemaRef
        :param list[str] required: list of required property names
        :param items: type of array items, only valid if `type` is ``array``
        :type items: Schema or SchemaRef
        :param default: only valid when insider another ``Schema``\\ 's ``properties``;
            the default value of this property if it is not provided, must conform to the type of this Schema
        :param read_only: only valid when insider another ``Schema``\\ 's ``properties``;
            declares the property as read only - it must only be sent as part of responses, never in requests
        """
        super(Schema, self).__init__(**extra)
        if required is True or required is False:
            # common error
            raise AssertionError("the `required` attribute of schema must be an "
                                 "array of required property names, not a boolean!")
        assert type, "type is required!"
        self.title = title
        self.description = description
        self.required = filter_none(required)
        self.type = type
        self.properties = filter_none(properties)
        self.additional_properties = additional_properties
        self.format = format
        self.enum = enum
        self.pattern = pattern
        self.items_ = items
        self.read_only = read_only
        self.default = default
        self._insert_extras__()
        if (properties or (additional_properties is not None)) and type != TYPE_OBJECT:
            raise AssertionError("only object Schema can have properties")
        _check_type(type, format, enum, pattern, items, self.__class__)

    def _remove_read_only(self):
        # readOnly is only valid for Schemas inside another Schema's properties;
        # when placing Schema elsewhere we must take care to remove the readOnly flag
        self.pop('readOnly', '')


class _Ref(SwaggerDict):
    ref: Any
    ref_name_re: Any = re.compile(r"#/(?P<scope>.+)/(?P<name>[^/]+)$")

    def __init__(self, resolver: Any, name: Any, scope: Any, expected_type: Any, ignore_unresolved: bool = False) -> None:
        """Base class for all reference types. A reference object has only one property, ``$ref``, which must be a JSON
        reference to a valid object in the specification, e.g. ``#/definitions/Article`` to refer to an article model.

        :param .ReferenceResolver resolver: component resolver which must contain the referneced object
        :param str name: referenced object name, e.g. "Article"
        :param str scope: reference scope, e.g. "definitions"
        :param type[.SwaggerDict] expected_type: the expected type that will be asserted on the object found in resolver
        :param bool ignore_unresolved: do not throw if the referenced object does not exist
        """
        super(_Ref, self).__init__()
        assert not type(self) == _Ref, "do not instantiate _Ref directly"
        ref_name = "#/{scope}/{name}".format(scope=scope, name=name)
        if not ignore_unresolved:
            obj = resolver.get(name, scope)
            assert isinstance(obj, expected_type), ref_name + " is a {actual}, not a {expected}" \
                .format(actual=type(obj).__name__, expected=expected_type.__name__)
        self.ref = ref_name

    def resolve(self, resolver: Any) -> Any:
        """Get the object targeted by this reference from the given component resolver.

        :param .ReferenceResolver resolver: component resolver which must contain the referneced object
        :returns: the target object
        """
        ref_match = self.ref_name_re.match(self.ref)
        return resolver.get(ref_match.group('name'), scope=ref_match.group('scope'))

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "$ref":
            return super(_Ref, self).__setitem__(key, value)
        raise NotImplementedError("only $ref can be set on Reference objects (not %s)" % key)

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError("cannot delete property of Reference object")


class SchemaRef(_Ref):
    def __init__(self, resolver: Any, schema_name: Any, ignore_unresolved: bool = False) -> None:
        """Adds a reference to a named Schema defined in the ``#/definitions/`` object.

        :param .ReferenceResolver resolver: component resolver which must contain the definition
        :param str schema_name: schema name
        :param bool ignore_unresolved: do not throw if the referenced object does not exist
        """
        assert SCHEMA_DEFINITIONS in resolver.scopes
        super(SchemaRef, self).__init__(resolver, schema_name, SCHEMA_DEFINITIONS, Schema, ignore_unresolved)


Schema.OR_REF = (Schema, SchemaRef)


def resolve_ref(ref_or_obj: Any, resolver: Any) -> Any:
    """Resolve `ref_or_obj` if it is a reference type. Return it unchaged if not.

    :param ref_or_obj: object to derefernece
    :type ref_or_obj: SwaggerDict or _Ref
    :param resolver: component resolver which must contain the referenced object
    """
    if isinstance(ref_or_obj, _Ref):
        return ref_or_obj.resolve(resolver)
    return ref_or_obj


class Responses(SwaggerDict):
    default: Optional[Response]
    def __init__(self, responses: Dict[Union[str, int], Response], default: Optional[Response] = None, **extra: Any) -> None:
        """Describes the expected responses of an :class:`.Operation`.

        :param responses: mapping of status code to response definition
        :type responses: dict[str or int,Response]
        :param Response default: description of the response structure to expect if another status code is returned
        """
        super(Responses, self).__init__(**extra)
        for status, response in responses.items():
            if response is not None:  # pragma: no cover
                self[str(status)] = response
        self.default = default
        self._insert_extras__()


_SerializerOrClass = Union[Serializer, Type[Serializer]]
class Response(SwaggerDict):
    description: str
    schema: Optional[Union[_SchemaOrRef, _SerializerOrClass]]
    examples: Optional[Dict[str, Any]]
    def __init__(self, description: str, schema: Optional[Union[_SchemaOrRef, _SerializerOrClass]] = None, examples: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
        """Describes the structure of an operation's response.

        :param str description: response description
        :param schema: structure of the response body
        :type schema: Schema or SchemaRef or rest_framework.serializers.Serializer
            or type[rest_framework.serializers.Serializer]
        :param dict examples: example bodies mapped by mime type
        """
        super(Response, self).__init__(**extra)
        self.description = description
        self.schema = schema
        self.examples = examples
        self._insert_extras__()
        if schema and isinstance(schema, Schema):
            schema._remove_read_only()


class ReferenceResolver(object):
    """A mapping type intended for storing objects pointed at by Swagger Refs.
    Provides support and checks for different refernce scopes, e.g. 'definitions'.

    For example:

    ::

        > components = ReferenceResolver('definitions', 'parameters')
        > definitions = components.with_scope('definitions')
        > definitions.set('Article', Schema(...))
        > print(components)
        {'definitions': OrderedDict([('Article', Schema(...)]), 'parameters': OrderedDict()}
    """

    def __init__(self, *scopes: str, force_init: bool = False, **kwargs: Any) -> None:
        """
        :param str scopes: an enumeration of the valid scopes this resolver will contain
        """
        if not force_init:
            raise AssertionError(
                "Creating an instance of ReferenceResolver almost certainly won't do what you want it to do.\n"
                "See https://github.com/axnsan12/drf-yasg/issues/211, "
                "https://github.com/axnsan12/drf-yasg/issues/271, "
                "https://github.com/axnsan12/drf-yasg/issues/325.\n"
                "Pass `force_init=True` to override this."
            )

        self._objects = OrderedDict()
        self._force_scope = None
        for scope in scopes:
            assert isinstance(scope, str), "scope names must be strings"
            self._objects[scope] = OrderedDict()

    def with_scope(self, scope: str) -> ReferenceResolver:
        """Return a view into this :class:`.ReferenceResolver` whose scope is defaulted and forced to `scope`.

        :param str scope: target scope, must be in this resolver's `scopes`
        :return: the bound resolver
        :rtype: .ReferenceResolver
        """
        assert scope in self.scopes, "unknown scope %s" % scope
        ret = ReferenceResolver(force_init=True)
        ret._objects = self._objects
        ret._force_scope = scope
        return ret

    def _check_scope(self, scope):
        real_scope = self._force_scope or scope
        if scope is not None:
            assert not self._force_scope or scope == self._force_scope, "cannot overrride forced scope"
        assert real_scope and real_scope in self._objects, "invalid scope %s" % scope
        return real_scope

    def set(self, name: str, obj: Any, scope: Optional[str] = None) -> None:
        """Set an object in the given scope, raise an error if it already exists.

        :param str name: reference name
        :param obj: referenced object
        :param str scope: reference scope
        """
        scope = self._check_scope(scope)
        assert obj is not None, "referenced objects cannot be None/null"
        assert name not in self._objects[scope], "#/%s/%s already exists" % (scope, name)
        self._objects[scope][name] = obj

    def setdefault(self, name: str, maker: Callable[[], Any], scope: Optional[str] = None) -> Any:
        """Set an object in the given scope only if it does not exist.

        :param str name: reference name
        :param function maker: object factory, called only if necessary
        :param str scope: reference scope
        """
        scope = self._check_scope(scope)
        assert callable(maker), "setdefault expects a callable, not %s" % type(maker).__name__
        ret = self.getdefault(name, None, scope)
        if ret is None:
            ret = maker()
            value = self.getdefault(name, None, scope)
            assert ret is not None, "maker returned None; referenced objects cannot be None/null"
            if value is None:
                self.set(name, ret, scope)
            elif value != ret:
                logger.debug("during setdefault, maker for %s inserted a value and returned a different value", name)
                ret = value

        return ret

    def get(self, name: str, scope: Optional[str] = None) -> Any:
        """Get an object from the given scope, raise an error if it does not exist.

        :param str name: reference name
        :param str scope: reference scope
        :return: the object
        """
        scope = self._check_scope(scope)
        assert name in self._objects[scope], "#/%s/%s is not defined" % (scope, name)
        return self._objects[scope][name]

    def getdefault(self, name: str, default: Optional[Any] = None, scope: Optional[str] = None) -> Any:
        """Get an object from the given scope or a default value if it does not exist.

        :param str name: reference name
        :param default: the default value
        :param str scope: reference scope
        :return: the object or `default`
        """
        scope = self._check_scope(scope)
        return self._objects[scope].get(name, default)

    def has(self, name: str, scope: Optional[str] = None) -> bool:
        """Check if an object exists in the given scope.

        :param str name: reference name
        :param str scope: reference scope
        :return: True if the object exists
        :rtype: bool
        """
        scope = self._check_scope(scope)
        return name in self._objects[scope]

    def __iter__(self) -> Iterator[Any]:
        if self._force_scope:
            return iter(self._objects[self._force_scope])
        return iter(self._objects)

    @property
    def scopes(self) -> List[str]:
        if self._force_scope:
            return [self._force_scope]
        return list(self._objects.keys())

    # act as mapping
    def keys(self) -> KeysView:
        if self._force_scope:
            return self._objects[self._force_scope].keys()
        return self._objects.keys()

    def __getitem__(self, item: str) -> Any:
        if self._force_scope:
            return self._objects[self._force_scope][item]
        return self._objects[item]

    def __str__(self):
        return str(dict(self))
