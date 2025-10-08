try:
    import truststore
    truststore.inject_into_ssl()  # use Windows cert store (includes Zscaler)
except Exception:
    pass
