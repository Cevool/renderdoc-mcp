"""
RenderDoc MCP Bridge Extension
Provides instance-isolated file IPC for external MCP clients.
"""

# Global state
_context = None
_server = None
_version = ""

# Try to import qrenderdoc for UI integration (only available in RenderDoc)
try:
    import qrenderdoc as qrd

    _has_qrenderdoc = True
except ImportError:
    _has_qrenderdoc = False


def register(version, ctx):
    """
    Called when extension is loaded.

    Args:
        version: RenderDoc version string (e.g., "1.20")
        ctx: CaptureContext handle
    """
    global _context, _server, _version
    from . import socket_server, request_handler, renderdoc_facade
    if _server is not None:
        _server.stop()
        _server = None
    _version = version
    _context = ctx

    # Create facade and handler
    facade = renderdoc_facade.RenderDocFacade(ctx)
    handler = request_handler.RequestHandler(facade)

    def status_provider():
        loaded = ctx.IsCaptureLoaded()
        return {"loaded": loaded, "filename": ctx.GetCaptureFilename() if loaded else None}

    _server = socket_server.MCPBridgeServer(
        handler=handler, version=version, status_provider=status_provider
    )
    _server.start()

    # Register menu item if UI is available
    if _has_qrenderdoc:
        try:
            ctx.Extensions().RegisterWindowMenu(
                qrd.WindowMenu.Tools, ["MCP Bridge", "Status"], _show_status
            )
        except Exception as e:
            print("[MCP Bridge] Could not register menu: %s" % str(e))

    print("[MCP Bridge] Extension loaded (RenderDoc %s)" % version)


def unregister():
    """Called when extension is unloaded"""
    global _server
    if _server:
        _server.stop()
        _server = None
    print("[MCP Bridge] Extension unloaded")


def _show_status(ctx, data):
    """Show status dialog"""
    if _server and _server.is_running():
        ctx.Extensions().MessageDialog(
            "Instance: %s\nIPC: %s" % (_server.bridge.instance_id, _server.bridge.directory),
            "MCP Bridge Status"
        )
    else:
        ctx.Extensions().ErrorDialog("MCP Bridge is not running", "MCP Bridge Status")
