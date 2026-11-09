from __future__ import annotations


class MissingConnectorOption(Exception):
    def __init__(self, option, connector):
        super().__init__(f"{option} is required to connect to {connector}")


class UnknownConnectorOption(MissingConnectorOption):
    """URI options the connector does not read, refused rather than dropped.

    It subclasses :class:`MissingConnectorOption`, so a caller that handles a URI
    whose options do not make up a usable connection catches both.
    """

    def __init__(self, options, connector, detail):
        Exception.__init__(
            self,
            f"{connector} does not recognise the URI option(s) "
            f"{', '.join(options)}. {detail}",
        )


class InvalidBlobTableError(Exception):
    def __init__(self, source):
        super().__init__(
            f"Invalid source table for: {source}. "
            "Ensure that the table is in the format {bucket-name}/{file glob}"
        )
