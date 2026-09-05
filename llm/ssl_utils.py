"""
SSL 上下文工具。

提供 `get_ssl_context()`，优先使用 certifi 的 CA 证书文件构建
SSLContext，避免在 macOS / 自定义 Python 构建上出现证书验证失败。
"""
from typing import Optional


def get_ssl_context():
    """返回一个 SSLContext（如果可用），否则返回 None。

    尝试按优先级：
    1. 使用 certifi.where() 提供的 CA 文件创建 context
    2. 回退为系统默认 context
    3. 出错时返回 None
    """
    try:
        import ssl
        try:
            import certifi
            cafile = certifi.where()
        except Exception:
            cafile = None

        if cafile:
            try:
                return ssl.create_default_context(cafile=cafile)
            except Exception:
                pass

        try:
            return ssl.create_default_context()
        except Exception:
            return None
    except Exception:
        return None
