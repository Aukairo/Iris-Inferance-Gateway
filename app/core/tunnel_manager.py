import os
import sys
import platform
import subprocess
import threading
import logging
import re
from typing import Optional, Dict, Any
import httpx
from app.config.config import settings

logger = logging.getLogger("mlx_server.tunnel")

class TunnelManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = "stopped"  # stopped, downloading, starting, active, error
        self.url: Optional[str] = None
        self.error_message: Optional[str] = None
        self.download_progress = 0
        self.lock = threading.RLock()  # Use RLock to allow re-entrant locking
        
        # Determine path to save cloudflared
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.bin_dir = os.path.join(self.base_dir, "bin")
        self.binary_name = "cloudflared"
        self.binary_path = os.path.join(self.bin_dir, self.binary_name)

    def _asset_name(self) -> Optional[str]:
        """
        Return the cloudflared release asset filename for this platform/arch.
        macOS assets are now shipped as .tgz archives (since ~2024).
        Linux assets are still raw ELF binaries.
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        is_arm = "arm" in machine or "aarch" in machine

        if system == "darwin":
            return "cloudflared-darwin-arm64.tgz" if is_arm else "cloudflared-darwin-amd64.tgz"
        if system == "linux":
            return "cloudflared-linux-arm64" if is_arm else "cloudflared-linux-amd64"
        return None

    def _resolve_download_url(self, asset_name: str) -> str:
        """
        Ask the GitHub API for the exact CDN URL of the latest release asset.
        Falls back to the /releases/latest/download/ redirect URL if the API
        is unavailable or the asset cannot be found in the response.
        """
        fallback = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset_name}"
        try:
            resp = httpx.get(
                "https://api.github.com/repos/cloudflare/cloudflared/releases/latest",
                headers={"User-Agent": "llm-server/1.0", "Accept": "application/vnd.github+json"},
                timeout=15,
                follow_redirects=True,
            )
            if resp.status_code == 200:
                for asset in resp.json().get("assets", []):
                    if asset.get("name") == asset_name:
                        url = asset["browser_download_url"]
                        logger.info(f"Resolved cloudflared download URL via GitHub API: {url}")
                        return url
            logger.warning("cloudflared asset not found in GitHub API response, using fallback URL.")
        except Exception as exc:
            logger.warning(f"GitHub API lookup failed ({exc}), using fallback URL.")
        return fallback

    def _extract_binary(self, archive_path: str) -> bool:
        """
        Extract the cloudflared binary from a .tgz archive into self.binary_path.
        The archive contains a single executable named 'cloudflared'.
        """
        import tarfile
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Find the 'cloudflared' member (may be at root or in a subdir)
                member = next(
                    (m for m in tar.getmembers() if m.name.split("/")[-1] == "cloudflared"),
                    None
                )
                if member is None:
                    self.error_message = "cloudflared binary not found inside the downloaded archive."
                    self.status = "error"
                    return False

                member.name = "cloudflared"  # flatten any subdirectory prefix
                tar.extract(member, path=self.bin_dir)
            return True
        except Exception as e:
            self.error_message = f"Archive extraction failed: {str(e)}"
            self.status = "error"
            logger.error(self.error_message)
            return False

    def download_binary(self) -> bool:
        asset_name = self._asset_name()
        if not asset_name:
            self.error_message = f"Unsupported platform: {platform.system()} {platform.machine()}"
            self.status = "error"
            return False

        os.makedirs(self.bin_dir, exist_ok=True)
        url = self._resolve_download_url(asset_name)
        is_archive = asset_name.endswith(".tgz") or asset_name.endswith(".tar.gz")
        download_path = self.binary_path + (".tgz" if is_archive else "")

        try:
            logger.info(f"Downloading cloudflared from {url}...")
            # httpx handles redirects and SSL (certifi) automatically.
            with httpx.Client(follow_redirects=True, timeout=300) as client:
                with client.stream("GET", url, headers={"User-Agent": "llm-server/1.0"}) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(download_path, "wb") as f:
                        for chunk in response.iter_bytes(65_536):
                            downloaded += len(chunk)
                            f.write(chunk)
                            if total_size:
                                self.download_progress = int((downloaded / total_size) * 100)

            if is_archive:
                logger.info("Extracting cloudflared binary from archive...")
                success = self._extract_binary(download_path)
                # Always remove the archive after extraction attempt
                try:
                    os.remove(download_path)
                except Exception:
                    pass
                if not success:
                    return False

            os.chmod(self.binary_path, 0o755)
            logger.info("cloudflared downloaded successfully.")
            return True

        except Exception as e:
            self.error_message = f"Download failed: {str(e)}"
            self.status = "error"
            logger.error(self.error_message)
            for p in (download_path, self.binary_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            return False



    def start_tunnel(self) -> Dict[str, Any]:
        with self.lock:
            if self.status in ["downloading", "starting", "active"]:
                return self.get_status()
                
            self.status = "starting"
            self.url = None
            self.error_message = None
            self.download_progress = 0
            
        # Start background thread to handle download/start
        threading.Thread(target=self._run_tunnel_thread, daemon=True).start()
        return self.get_status()

    def _run_tunnel_thread(self):
        # 1. Check/Download binary
        if not os.path.exists(self.binary_path):
            with self.lock:
                self.status = "downloading"
            
            success = self.download_binary()
            if not success:
                return

        # 2. Run cloudflared
        with self.lock:
            self.status = "starting"
            
        try:
            port = settings.port
            host = settings.host
            local_url = f"http://{host}:{port}"
            
            logger.info(f"Starting cloudflared tunnel forwarding to {local_url}...")
            # Quick tunnels are launched via: cloudflared tunnel --url <local_url>
            self.process = subprocess.Popen(
                [self.binary_path, "tunnel", "--url", local_url],
                stdin=subprocess.DEVNULL,   # Prevent process from waiting for input
                stdout=subprocess.DEVNULL,  # stdout is unused; DEVNULL prevents pipe buffer deadlock
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Read stderr in a background loop to find the URL
            # Cloudflare logs tunnel creation progress to stderr
            url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
            
            while True:
                line = self.process.stderr.readline()
                if not line:
                    break
                
                # Check for trycloudflare URL
                match = url_pattern.search(line)
                if match:
                    with self.lock:
                        self.url = match.group(0)
                        self.status = "active"
                    logger.info(f"Public tunnel active! URL: {self.url}")
                    break
                    
                # If process died early
                if self.process.poll() is not None:
                    break

            # Start monitoring process death in another thread
            threading.Thread(target=self._monitor_process, daemon=True).start()

        except Exception as e:
            with self.lock:
                self.status = "error"
                self.error_message = f"Failed to launch tunnel process: {str(e)}"
            logger.error(self.error_message)

    def _monitor_process(self):
        if not self.process:
            return
        try:
            self.process.wait()
        except Exception:
            pass
        
        # Read remaining stderr for error context (with short timeout to avoid blocking)
        stderr_output = ""
        try:
            if self.process and self.process.stderr:
                self.process.stderr.close()
        except Exception:
            pass
        
        with self.lock:
            if self.status not in ("stopped", "error"):
                self.status = "error"
                self.error_message = "Tunnel process exited unexpectedly."
                logger.error(f"Tunnel process terminated: {self.error_message}")
            self.process = None
            self.url = None

    def stop_tunnel(self) -> Dict[str, Any]:
        """Stop the tunnel with timeout protection."""
        logger.info("[tunnel] stop_tunnel() called")
        
        proc = None
        try:
            # Acquire lock with timeout to prevent indefinite hangs
            acquired = self.lock.acquire(timeout=1.0)
            if not acquired:
                logger.error("[tunnel] CRITICAL: Failed to acquire lock - returning")
                return {"status": "unknown", "error": "Lock timeout"}
            
            try:
                if self.status == "stopped":
                    logger.info("[tunnel] Already stopped")
                    return self.get_status()
                
                logger.info(f"[tunnel] Stopping (status: {self.status})")
                self.status = "stopped"
                self.url = None
                proc = self.process
                self.process = None
            finally:
                self.lock.release()
        except Exception as e:
            logger.error(f"[tunnel] Lock error: {e}")
            return {"status": "error", "error": str(e)}
        
        # Terminate outside lock
        if proc:
            logger.info("[tunnel] Terminating process")
            try:
                # Close pipes
                for pipe in [proc.stderr, proc.stdout, proc.stdin]:
                    if pipe:
                        try:
                            pipe.close()
                        except Exception:
                            pass
                
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                    logger.info("[tunnel] Terminated cleanly")
                except subprocess.TimeoutExpired:
                    logger.warning("[tunnel] Timeout, killing")
                    proc.kill()
                    try:
                        proc.wait(timeout=1)
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"[tunnel] Termination error: {e}")
                try:
                    proc.kill()
                except Exception:
                    pass
        
        logger.info("[tunnel] stop_tunnel() done")
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "url": self.url,
                "error": self.error_message,
                "progress": self.download_progress,
                "binary_exists": os.path.exists(self.binary_path),
            }

tunnel_manager = TunnelManager()
