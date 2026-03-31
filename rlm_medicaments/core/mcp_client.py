"""Client MCP data.gouv.fr pour le téléchargement des ressources BDPM.

Utilise le protocole MCP (Model Context Protocol) via l'endpoint
https://mcp.data.gouv.fr/mcp pour chercher et accéder aux données.
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

MCP_ENDPOINT = "https://mcp.data.gouv.fr/mcp"

# Dataset Défi iDoc Santé — contient les fichiers BDPM concrets (zip + csv)
DATASET_ID = "62694159aefe65020a033bdc"

# Resource IDs within the dataset
RESOURCE_IDS = {
    "CIS_bdpm_zip": "fecf69dd-ca9f-4902-95dd-4e0ec6ab92f0",  # CIS_bdpm_officielle.zip
    "CIS_RCP_zip": "bdbe2367-1898-4848-ac85-6fe58a1bdf68",    # CIS_RCP.zip (153 MB)
    "CIS_Pathologie": "f549a488-d0fb-4ded-8fe0-1bbd6f5653bd", # CIS_Pathologie.csv
}

# Files to extract from CIS_bdpm_officielle.zip
BDPM_FILES = {
    "CIS_bdpm": "CIS_bdpm.csv",
    "CIS_COMPO_bdpm": "CIS_COMPO_bdpm.csv",
    "CIS_CIP_bdpm": "CIS_CIP_bdpm.csv",
    "CIS_GENER_bdpm": "CIS_GENER_bdpm.csv",
    "CIS_HAS_SMR_bdpm": "CIS_HAS_SMR_bdpm.csv",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── MCP Protocol ─────────────────────────────────────────────────────────────


def mcp_call(tool_name: str, arguments: dict, retries: int = 3) -> dict:
    """Call a tool on the MCP data.gouv.fr server via JSON-RPC over HTTP+SSE.

    Retries on transient errors (502, timeout).
    """
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }).encode()

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                MCP_ENDPOINT,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                for line in body.splitlines():
                    if line.startswith("data:"):
                        msg = json.loads(line[5:].strip())
                        result = msg.get("result", {})
                        if result.get("isError"):
                            error_text = result.get("content", [{}])[0].get("text", "Unknown MCP error")
                            raise RuntimeError(f"MCP error: {error_text}")
                        return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt < retries - 1:
                import time
                time.sleep(2 * (attempt + 1))
                continue
            break

    raise RuntimeError(f"MCP server unreachable after {retries} attempts: {last_error}")


def mcp_get_text(tool_name: str, arguments: dict) -> str:
    """Call an MCP tool and return the text content."""
    result = mcp_call(tool_name, arguments)
    content = result.get("content", [])
    if content:
        return content[0].get("text", "")
    return ""


def mcp_search_datasets(query: str, page_size: int = 5) -> str:
    """Search datasets via MCP."""
    return mcp_get_text("search_datasets", {"query": query, "page_size": page_size})


def mcp_list_resources(dataset_id: str) -> str:
    """List resources of a dataset via MCP."""
    return mcp_get_text("list_dataset_resources", {"dataset_id": dataset_id})


def mcp_get_resource_info(resource_id: str) -> str:
    """Get info about a specific resource via MCP."""
    return mcp_get_text("get_resource_info", {"resource_id": resource_id})


def mcp_query_data(resource_id: str, question: str, page: int = 1,
                   page_size: int = 20, **kwargs) -> str:
    """Query tabular data from a resource via MCP Tabular API."""
    args = {
        "question": question,
        "resource_id": resource_id,
        "page": page,
        "page_size": page_size,
        **kwargs,
    }
    return mcp_get_text("query_resource_data", args)


# ── Download helpers ─────────────────────────────────────────────────────────


def _download_url(url: str, timeout: int = 120) -> bytes:
    """Download raw bytes from a URL."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


FALLBACK_ZIP_URL = (
    "https://static.data.gouv.fr/resources/"
    "base-de-donnees-publique-des-medicaments-defi-idoc-sante/"
    "20220502-154759/cis-bdpm-officielle.zip"
)


def _resolve_zip_url(on_status=None) -> str:
    """Resolve the download URL for CIS_bdpm_officielle.zip via MCP, with fallback."""
    try:
        if on_status:
            on_status("Interrogation du MCP data.gouv.fr...")
        info_text = mcp_get_resource_info(RESOURCE_IDS["CIS_bdpm_zip"])
        for line in info_text.splitlines():
            if line.strip().startswith("URL:"):
                url = line.split("URL:", 1)[1].strip()
                if url:
                    return url
    except Exception:
        if on_status:
            on_status("MCP indisponible — utilisation de l'URL directe.")
    return FALLBACK_ZIP_URL


def download_bdpm_zip(on_status=None) -> dict[str, Path]:
    """Download and extract CIS_bdpm_officielle.zip.

    Uses MCP to discover the URL, falls back to known static URL.
    Returns dict of key -> local extracted file path.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    all_cached = all(
        (DATA_DIR / filename).exists() for filename in BDPM_FILES.values()
    )
    if all_cached:
        if on_status:
            on_status("Fichiers BDPM déjà en cache local.")
        return {key: DATA_DIR / filename for key, filename in BDPM_FILES.items()}

    zip_url = _resolve_zip_url(on_status)

    if on_status:
        on_status("Téléchargement de CIS_bdpm_officielle.zip (3.3 Mo)...")

    zip_data = _download_url(zip_url)

    if on_status:
        on_status("Extraction des fichiers...")

    extracted = {}
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for key, filename in BDPM_FILES.items():
            if filename in zf.namelist():
                content = zf.read(filename)
                dest = DATA_DIR / filename
                dest.write_bytes(content)
                extracted[key] = dest
                if on_status:
                    on_status(f"  Extrait: {filename} ({len(content):,} octets)")

    return extracted


def download_bdpm_files(on_status=None) -> dict[str, Path]:
    """Download all required BDPM files. Main entry point.

    Uses MCP to discover and download data from data.gouv.fr.
    Returns dict of name -> local file path.
    """
    return download_bdpm_zip(on_status=on_status)


def clear_cache():
    """Remove all cached BDPM files."""
    if DATA_DIR.exists():
        for f in DATA_DIR.iterdir():
            if f.is_file():
                f.unlink()
