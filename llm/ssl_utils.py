"""
SSL 上下文工具。

提供 `get_ssl_context()`，优先使用 certifi 的 CA 证书文件构建
SSLContext，避免在 macOS / 自定义 Python 构建上出现证书验证失败。
"""
import os
import ssl


def get_ssl_context():
    """Use an explicit/system CA bundle without disabling certificate validation."""
    explicit = os.environ.get("SSL_CERT_FILE")
    if explicit:
        if not os.path.isfile(explicit):
            raise ValueError(f"SSL_CERT_FILE 指向不存在的文件: {explicit}")
        return ssl.create_default_context(cafile=explicit)

    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        pass

    default = ssl.get_default_verify_paths()
    for candidate in (default.cafile, "/etc/ssl/cert.pem"):
        if candidate and os.path.isfile(candidate):
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()
