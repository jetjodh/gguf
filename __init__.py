try:
    import comfy.utils
except ImportError:
    pass
else:
    from .pig import NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = {k:v.TITLE for k,v in NODE_CLASS_MAPPINGS.items()}
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
