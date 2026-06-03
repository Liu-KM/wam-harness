"""Compatibility policy client for DreamZero DROID sim evaluation.

DreamZero's upstream policy client assumes a newer ``websockets`` sync-client
signature, while IsaacSim currently pins the simulator environment to
``websockets==12.0``.  Keep this shim narrow: it mirrors the upstream client and
falls back to the older connect signature when ping arguments are unsupported.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from openpi_client import msgpack_numpy
from openpi_client.base_policy import BasePolicy
from typing_extensions import override
import websockets.sync.client

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


def _connect(uri: str) -> websockets.sync.client.ClientConnection:
    kwargs = {
        "compression": None,
        "max_size": None,
        "ping_interval": PING_INTERVAL_SECS,
        "ping_timeout": PING_TIMEOUT_SECS,
    }
    try:
        return websockets.sync.client.connect(uri, **kwargs)
    except TypeError as exc:
        if "ping_interval" not in str(exc) and "ping_timeout" not in str(exc):
            raise
        kwargs.pop("ping_interval")
        kwargs.pop("ping_timeout")
        return websockets.sync.client.connect(uri, **kwargs)


class WebsocketClientPolicy(BasePolicy):
    """Policy interface backed by DreamZero's WebSocket server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for server at %s...", self._uri)
        try:
            conn = _connect(self._uri)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")

        self._uri = "wss://" + self._uri.split("//")[1]
        conn = _connect(self._uri)
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, reset_info: Dict) -> None:
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        self._ws.send(data)
        return self._ws.recv()
