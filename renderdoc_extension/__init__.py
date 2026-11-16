"""
RenderDoc MCP Bridge Extension
Provides instance-isolated file IPC for external MCP clients.
"""

# Global state
_context = None
_server = None
_version = ""
_facade = None
_capture_viewer = None

# Try to import qrenderdoc for UI integration (only available in RenderDoc)
try:
    import qrenderdoc as qrd

    _has_qrenderdoc = True
except ImportError:
    _has_qrenderdoc = False


if _has_qrenderdoc:
    class _CaptureViewer(qrd.CaptureViewer):
        """Capture load/close invalidates MCP-owned replay edits and evidence sessions."""

        def __init__(self, facade):
            super(_CaptureViewer, self).__init__()
            self.facade = facade

        def OnCaptureLoaded(self):
            self.facade.capture_changed()

        def OnCaptureClosed(self):
            self.facade.capture_changed()

        def OnSelectedEventChanged(self, event):
            pass

        def OnEventChanged(self, event):
            pass


def register(version, ctx):
    """
    Called when extension is loaded.

    Args:
        version: RenderDoc version string (e.g., "1.20")
        ctx: CaptureContext handle
    """
    global _context, _server, _version, _facade, _capture_viewer
    from . import socket_server, request_handler, renderdoc_facade
    if _server is not None:
        unregister()
    _version = version
    _context = ctx

    # Create facade and handler
    facade = renderdoc_facade.RenderDocFacade(ctx)
    _facade = facade
    if _has_qrenderdoc:
        try:
            _capture_viewer = _CaptureViewer(facade)
            ctx.AddCaptureViewer(_capture_viewer)
        except Exception as e:
            _capture_viewer = None
            print("[MCP Bridge] Capture change notifications unavailable: %s" % str(e))
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
    global _server, _facade, _capture_viewer
    if _facade:
        _facade.shutdown()  # Revert MCP shader replacements; nothing else would own them.
        _facade = None
    if _capture_viewer:
        try:
            _context.RemoveCaptureViewer(_capture_viewer)
        except Exception as e:
            print("[MCP Bridge] Could not remove capture viewer: %s" % str(e))
        _capture_viewer = None
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
