"""Parse and validate possibly qualified table names."""

from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

from omniload.error import ValidationError


@dataclass(frozen=True)
class TableName:
    """A right-aligned table name with optional catalog and schema components."""

    catalog: str | None
    schema: str | None
    table: str

    def qualified_schema(self) -> str | None:
        """Return the schema, prefixed by the catalog when both are present."""
        if self.catalog and self.schema:
            return f"{self.catalog}.{self.schema}"
        return self.schema

    def __str__(self) -> str:
        """Join the present components with dots."""
        return ".".join(
            component
            for component in (self.catalog, self.schema, self.table)
            if component is not None
        )


@dataclass(frozen=True)
class Defaults:
    """Fallback catalog and schema values for an unqualified name."""

    catalog: str | None = None
    schema: str | None = None


SplitKind = Literal["qualified", "prefix", "opaque"]


@dataclass(frozen=True)
class Capability:
    """Describe and parse the table-name shape accepted by one platform."""

    platform: str
    min_components: int
    max_components: int
    labels: tuple[str, str, str]
    format_desc: str
    unbounded: bool = False
    catalog_hint: str | None = None
    split_kind: SplitKind = "qualified"

    def _parts(self, raw: str) -> list[str]:
        if self.split_kind == "opaque":
            return [raw.strip()]
        if self.split_kind == "prefix":
            return _prefix_split(raw)
        return split(raw)

    def error(self, raw: str, hint: str | None = None) -> ValidationError:
        """Build the platform-specific validation error for a raw name."""
        message = (
            f"The '{self.platform}' table name must be in the format "
            f"{self.format_desc}; got {raw!r}."
        )
        detail = hint or self.catalog_hint
        if detail:
            message = f"{message} {detail}"
        return ValidationError(message)

    def unresolved_schema_error(self, raw: str) -> ValidationError:
        """Build the error for a name that neither carries nor defaults its schema."""
        label = self.labels[1] or "schema"
        return ValidationError(
            f"The '{self.platform}' table name {raw!r} names no {label}, and the "
            f"connection URI supplies no default. Write it as "
            f"<{label}>.<{self.labels[2]}>, or name the {label} in the URI."
        )

    def check(self, raw: str) -> None:
        """Validate component content and count for this capability."""
        if self.split_kind != "opaque" and _is_whole_backtick_path(raw):
            raise ValidationError(
                f"The '{self.platform}' table name {raw!r} wraps the whole path in "
                "backticks. Quote each component separately, for example "
                "`catalog`.`schema`.`table`. For a single identifier that contains a "
                "dot, quote it with double quotes or square brackets instead."
            )

        parts = self._parts(raw)
        if any(not part.strip() for part in parts):
            raise self.error(raw)

        component_count = len(parts)
        if component_count < self.min_components or (
            not self.unbounded and component_count > self.max_components
        ):
            raise self.error(raw)

    def parse(self, raw: str, defaults: Defaults | None = None) -> TableName:
        """Parse a name by right-aligning its components and applying defaults."""
        self.check(raw)
        defaults = defaults or Defaults()
        parts = self._parts(raw)

        if self.split_kind == "opaque":
            return TableName(defaults.catalog, defaults.schema, parts[0])

        parsed = TableName(defaults.catalog, defaults.schema, parts[-1])
        if len(parts) >= 2:
            parsed = TableName(parsed.catalog, parts[-2], parsed.table)
        if len(parts) >= 3:
            parsed = TableName(".".join(parts[:-2]), parsed.schema, parsed.table)
        return parsed


def _normalize_part(part: str) -> str:
    part = part.strip()
    if len(part) < 2:
        return part

    opener = part[0]
    closer = part[-1]
    if opener == "[" and closer == "]":
        return part[1:-1].replace("]]", "]")
    if opener == closer == '"':
        return part[1:-1].replace('""', '"')
    if opener == closer == "`":
        return part[1:-1].replace("``", "`")
    return part


def split(raw: str) -> list[str]:
    """Split on unquoted dots and unquote each identifier component."""
    value = raw.strip()
    parts: list[str] = []
    current: list[str] = []
    closer: str | None = None

    index = 0
    while index < len(value):
        char = value[index]
        if closer is not None:
            current.append(char)
            if char == closer:
                if index + 1 < len(value) and value[index + 1] == closer:
                    index += 1
                    current.append(value[index])
                else:
                    closer = None
            index += 1
            continue

        if char in {"[", '"', "`"}:
            closer = "]" if char == "[" else char
            current.append(char)
        elif char == ".":
            parts.append(_normalize_part("".join(current)))
            current = []
        else:
            current.append(char)
        index += 1

    if closer is not None:
        raise ValidationError(f"Unterminated quoted identifier in table name {raw!r}.")

    parts.append(_normalize_part("".join(current)))
    return parts


def _prefix_split(raw: str) -> list[str]:
    """Split once on an unquoted dot and leave the remainder opaque."""
    value = raw.strip()
    closer: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if closer is not None:
            if char == closer:
                if index + 1 < len(value) and value[index + 1] == closer:
                    index += 1
                else:
                    closer = None
            index += 1
            continue

        if char in {"[", '"', "`"}:
            closer = "]" if char == "[" else char
        elif char == ".":
            return [
                _normalize_part(value[:index]),
                value[index + 1 :].strip(),
            ]
        index += 1

    if closer is not None:
        raise ValidationError(f"Unterminated quoted identifier in table name {raw!r}.")
    return [_normalize_part(value)]


def _is_whole_backtick_path(raw: str) -> bool:
    value = raw.strip()
    if len(value) < 3 or value[0] != "`" or "." not in value[1:-1]:
        return False

    index = 1
    while index < len(value):
        if value[index] == "`":
            if index + 1 < len(value) and value[index + 1] == "`":
                index += 2
                continue
            return index == len(value) - 1
        index += 1
    return False


def _format_desc(labels: tuple[str, ...], min_components: int) -> str:
    variants = []
    for count in range(min_components, len(labels) + 1):
        variants.append(".".join(f"<{label}>" for label in labels[-count:]))
    return " or ".join(variants)


def two_level(
    platform: str,
    catalog_hint: str | None = None,
    *,
    min_components: int = 2,
    schema_label: str = "schema",
    table_label: str = "table",
) -> Capability:
    """Return a capability for names ending in schema and table."""
    labels = (schema_label, table_label)
    return Capability(
        platform=platform,
        min_components=min_components,
        max_components=2,
        labels=("", *labels),
        format_desc=_format_desc(labels, min_components),
        catalog_hint=catalog_hint,
    )


def three_level(
    platform: str,
    catalog_hint: str | None = None,
    *,
    min_components: int = 2,
    catalog_label: str = "catalog",
    schema_label: str = "schema",
    table_label: str = "table",
) -> Capability:
    """Return a capability for names ending in catalog, schema, and table."""
    labels = (catalog_label, schema_label, table_label)
    return Capability(
        platform=platform,
        min_components=min_components,
        max_components=3,
        labels=labels,
        format_desc=_format_desc(labels, min_components),
        catalog_hint=catalog_hint,
    )


def one_level(
    platform: str,
    catalog_hint: str | None = None,
    *,
    table_label: str = "table",
) -> Capability:
    """Return a capability for exactly one table-name component."""
    return Capability(
        platform=platform,
        min_components=1,
        max_components=1,
        labels=("", "", table_label),
        format_desc=f"<{table_label}>",
        catalog_hint=catalog_hint,
    )


def opaque(platform: str, *, table_label: str = "table") -> Capability:
    """Return a capability whose entire value is an opaque table name."""
    return Capability(
        platform=platform,
        min_components=1,
        max_components=1,
        labels=("", "", table_label),
        format_desc=f"<{table_label}>",
        split_kind="opaque",
    )


def prefix_split(
    platform: str,
    labels: tuple[str, str],
    *,
    min_components: int = 2,
) -> Capability:
    """Return a first-component plus opaque-remainder capability."""
    return Capability(
        platform=platform,
        min_components=min_components,
        max_components=2,
        labels=("", *labels),
        format_desc=_format_desc(labels, min_components),
        split_kind="prefix",
    )


def uri_with_database(uri: str, database: str) -> str:
    """Replace a connection URI's first path component with a database name."""
    parsed = urlsplit(uri)
    if not parsed.scheme or not parsed.netloc:
        raise ValidationError(
            f"Cannot select database {database!r} from connection URI {uri!r}."  # noqa: S608
        )

    encoded_database = quote(database, safe="")
    path_components = parsed.path.lstrip("/").split("/") if parsed.path else []
    if path_components:
        path_components[0] = encoded_database
    else:
        path_components = [encoded_database]
    path = f"/{'/'.join(path_components)}"
    return urlunsplit(parsed._replace(path=path))
