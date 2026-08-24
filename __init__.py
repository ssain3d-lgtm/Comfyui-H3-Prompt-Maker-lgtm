from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import server_routes

# Serves web/h3_maker.js to ComfyUI's frontend as an extension.
WEB_DIRECTORY = "./web"

# Routes for the overlay app and its generate endpoint. A False return just
# means ComfyUI's server was not importable — the widget nodes still work.
server_routes.install()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
