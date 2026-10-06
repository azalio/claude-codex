from __future__ import annotations

import os
import ssl

import certifi
import truststore


def upstream_ssl_context() -> ssl.SSLContext:
    """Сохраняет публичные CA и системное доверие ОС; явный выбор CA имеет приоритет."""
    if cafile := os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context(cafile=cafile)
    if capath := os.environ.get("SSL_CERT_DIR"):
        return ssl.create_default_context(capath=capath)
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=certifi.where())
    context.load_default_certs()
    return context
