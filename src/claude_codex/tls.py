from __future__ import annotations

import os
import ssl

import certifi


def upstream_ssl_context() -> ssl.SSLContext:
    """Сохраняет публичные CA и добавляет системные; явный выбор CA имеет приоритет."""
    if cafile := os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context(cafile=cafile)
    if capath := os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context(capath=capath)
    context = ssl.create_default_context(cafile=certifi.where())
    context.load_default_certs()
    return context
