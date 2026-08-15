from dlt_filesystem.source.impl.remote import S3CompatibleSource


class R2Source(S3CompatibleSource):
    """
    Access files on Cloudflare R2.

    R2 is compatible with Amazon S3.
    https://github.com/panodata/omniload/issues/163
    """

    @property
    def fs_name(self) -> str:
        return "R2"

    @property
    def fs_protocol(self) -> str:
        return "r2"
