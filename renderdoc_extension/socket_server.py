"""Event-driven Qt adapter; filename retained for extension upgrade compatibility."""

from PySide2.QtCore import QFileSystemWatcher, QObject, QTimer

from .ipc import FileBridge

# Retry delays (ms) for a request file that is still locked; after these the five-second
# maintenance timer takes over, so a permanently locked file does not keep the UI busy.
RETRY_DELAYS = (2, 5, 10, 20, 50, 100, 200, 500)


class MCPBridgeServer(QObject):
    def __init__(self, handler, version="", status_provider=None, parent=None):
        super(MCPBridgeServer, self).__init__(parent)
        self.bridge = FileBridge(handler, version=version, status_provider=status_provider)
        self._watcher = QFileSystemWatcher(self)
        self._watcher.directoryChanged.connect(self._drain)
        self._timer = QTimer(self)
        self._timer.setInterval(5000)
        self._timer.timeout.connect(self._maintenance)
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.timeout.connect(self._drain)
        self._attempt = 0

    def start(self):
        self.bridge.start()
        self._watcher.addPath(self.bridge.requests)
        self._timer.start()
        self._drain()
        print("[MCP Bridge] Instance %s: %s" % (self.bridge.instance_id, self.bridge.directory))
        return True

    def stop(self):
        self._timer.stop()
        self._retry.stop()
        paths = self._watcher.directories()
        if paths:
            self._watcher.removePaths(paths)
        self.bridge.stop()

    def is_running(self):
        return self.bridge.running

    def _drain(self, *args):
        if self.bridge.drain():
            QTimer.singleShot(0, self._drain)
        if not self.bridge.contended:
            self._attempt = 0
        elif not self._retry.isActive() and self._attempt < len(RETRY_DELAYS):
            self._retry.start(RETRY_DELAYS[self._attempt])
            self._attempt += 1

    def _maintenance(self):
        self.bridge.refresh()
        self._attempt = 0
        self._drain()
