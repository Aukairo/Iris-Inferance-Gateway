import os
import unittest
import threading
from unittest.mock import patch, MagicMock
from app.core.tunnel_manager import TunnelManager

class TestTunnelManager(unittest.TestCase):
    def setUp(self):
        self.manager = TunnelManager()

    def test_initial_status(self):
        status = self.manager.get_status()
        self.assertEqual(status["status"], "stopped")
        self.assertIsNone(status["url"])
        self.assertIsNone(status["error"])

    @patch("platform.system")
    @patch("platform.machine")
    def test_get_download_url(self, mock_machine, mock_system):
        # Test macOS arm64
        mock_system.return_value = "Darwin"
        mock_machine.return_value = "arm64"
        asset = self.manager._asset_name()
        self.assertEqual(asset, "cloudflared-darwin-arm64.tgz")

        # Test macOS Intel
        mock_machine.return_value = "x86_64"
        asset = self.manager._asset_name()
        self.assertEqual(asset, "cloudflared-darwin-amd64.tgz")

        # Test Linux Intel
        mock_system.return_value = "Linux"
        mock_machine.return_value = "x86_64"
        asset = self.manager._asset_name()
        self.assertEqual(asset, "cloudflared-linux-amd64")

    @patch("app.core.tunnel_manager.threading.Thread")
    @patch("app.core.tunnel_manager.subprocess.Popen")
    @patch("app.core.tunnel_manager.os.path.exists")
    def test_start_and_stop_tunnel(self, mock_exists, mock_popen, mock_thread):
        mock_exists.return_value = True
        
        # Mock Thread to run target immediately in the same thread (skipping monitoring loop to avoid synchronous exit)
        class SyncThread:
            def __init__(self, target, daemon=True):
                self.target = target
            def start(self):
                if "run_tunnel" in self.target.__name__:
                    self.target()
                
        mock_thread.side_effect = SyncThread
        
        # Mock process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr.readline.side_effect = [
            "https://mock-tunnel.trycloudflare.com\n",
            ""
        ]
        mock_proc.stderr.read.return_value = ""
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        # Start tunnel
        print("1. Starting tunnel...")
        res = self.manager.start_tunnel()
        print("2. Started. Status:", self.manager.get_status())
        self.assertEqual(self.manager.get_status()["status"], "active")
        self.assertEqual(self.manager.get_status()["url"], "https://mock-tunnel.trycloudflare.com")

        # Mock active process
        mock_proc = MagicMock()
        self.manager.process = mock_proc
        
        # Stop tunnel
        print("3. Stopping tunnel...")
        stop_res = self.manager.stop_tunnel()
        print("4. Stopped.")
        self.assertEqual(stop_res["status"], "stopped")
        self.assertIsNone(stop_res["url"])
        mock_proc.terminate.assert_called_once()
