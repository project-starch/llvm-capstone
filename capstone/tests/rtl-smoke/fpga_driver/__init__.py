"""Headless Socket.IO driver for the CapliFive FPGA web console (rtl-smoke).

See README.md for the wire-up procedure and PROTOCOL.md for the protocol map.
The only external dependency is python-socketio (client extra). The mock server
(mock_server.py) additionally needs the server extra + aiohttp, for offline
dry-runs only.
"""
