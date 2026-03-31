"""Client MCP data.gouv.fr pour les rapports de la Cour des Comptes.

Utilise le protocole MCP (JSON-RPC sur HTTP+SSE) pour decouvrir dynamiquement
les datasets et leurs ressources, puis telecharge les fichiers.
Fallback sur URLs statiques si le MCP est indisponible.
"""

import io
import json
import re
import time
import urllib.request
import urllib.error
import zipfile
from html.parser import HTMLParser
from pathlib import Path

MCP_ENDPOINT = "https://mcp.data.gouv.fr/mcp"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Dataset IDs ──────────────────────────────────────────────────────────────

REPORTS_DATASET_ID = "57470e8688ee38574dd1b934"
RECOMMENDATIONS_DATASET_ID = "5b0c2635c751df68bfc675a2"

# Fallback URLs (used only if MCP is unreachable)
_FALLBACK_REPORT_ZIPS = {
    2013: "https://static.data.gouv.fr/resources/rapports-publies-par-la-cour-des-comptes/20160526-175931/Rapports_publics_de_la_Cour_et_metadonnees_-_2013.zip",
    2014: "https://static.data.gouv.fr/resources/rapports-publies-par-la-cour-des-comptes/20160526-175236/Rapports_publics_de_la_Cour_et_metadonnees_-_2014.zip",
    2015: "https://static.data.gouv.fr/resources/rapports-publies-par-la-cour-des-comptes/20160526-174842/Rapports_publics_de_la_Cour_et_metadonnees_-_2015.zip",
    2016: "https://static.data.gouv.fr/resources/rapports-publies-par-la-cour-des-comptes/20170622-221002/Rapports_publics_de_la_Cour_et_metadonnees_-_2016.zip",
}
_FALLBACK_RECOMMENDATIONS_URL = (
    "https://static.data.gouv.fr/resources/"
    "recommandations-publiees-par-la-cour-des-comptes-2015-mai-2018/"
    "20180528-175549/TdB_Cour_des_comptes.txt"
)


# ── MCP Protocol ─────────────────────────────────────────────────────────────


def mcp_call(tool_name: str, arguments: dict, retries: int = 3) -> dict:
    """Call a tool on the MCP data.gouv.fr server via JSON-RPC over HTTP+SSE."""
    payload = json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }).encode()

    last_error = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                MCP_ENDPOINT, data=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json, text/event-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
                for line in body.splitlines():
                    if line.startswith("data:"):
                        msg = json.loads(line[5:].strip())
                        result = msg.get("result", {})
                        if result.get("isError"):
                            raise RuntimeError(result.get("content", [{}])[0].get("text", "MCP error"))
                        return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"MCP unreachable after {retries} attempts: {last_error}")


def mcp_get_text(tool_name: str, arguments: dict) -> str:
    result = mcp_call(tool_name, arguments)
    content = result.get("content", [])
    return content[0].get("text", "") if content else ""


def mcp_list_resources(dataset_id: str) -> str:
    """List resources of a dataset via MCP."""
    return mcp_get_text("list_dataset_resources", {"dataset_id": dataset_id})


def _parse_resources_from_mcp(mcp_text: str) -> list[dict]:
    """Parse MCP list_dataset_resources output into structured dicts."""
    resources = []
    current: dict = {}
    for line in mcp_text.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s", line):
            if current:
                resources.append(current)
            current = {"title": re.sub(r"^\d+\.\s*", "", line)}
        elif line.startswith("URL:"):
            current["url"] = line.split("URL:", 1)[1].strip()
        elif line.startswith("Format:"):
            current["format"] = line.split("Format:", 1)[1].strip()
    if current:
        resources.append(current)
    return resources


# ── Dynamic resource discovery ───────────────────────────────────────────────


def discover_report_zips(on_status=None) -> dict[int, str]:
    """Discover report zip URLs via MCP. Returns {year: url}.

    Falls back to hardcoded URLs if MCP is unreachable.
    """
    try:
        if on_status:
            on_status("MCP: decouverte des ressources rapports...")
        text = mcp_list_resources(REPORTS_DATASET_ID)
        resources = _parse_resources_from_mcp(text)

        zips: dict[int, str] = {}
        for r in resources:
            url = r.get("url", "")
            if not url or r.get("format", "").lower() != "zip":
                continue
            # Extract year from title like "... (2016)" or URL
            year_match = re.search(r"(\d{4})", r.get("title", ""))
            if year_match:
                zips[int(year_match.group(1))] = url

        if zips:
            if on_status:
                on_status(f"MCP: {len(zips)} zips decouverts ({', '.join(str(y) for y in sorted(zips))})")
            return zips

    except Exception as e:
        if on_status:
            on_status(f"MCP indisponible ({e}) — fallback URLs statiques")

    return dict(_FALLBACK_REPORT_ZIPS)


def discover_recommendations_url(on_status=None) -> str:
    """Discover recommendations file URL via MCP.

    Falls back to hardcoded URL if MCP is unreachable.
    """
    try:
        if on_status:
            on_status("MCP: decouverte des ressources recommandations...")
        text = mcp_list_resources(RECOMMENDATIONS_DATASET_ID)
        resources = _parse_resources_from_mcp(text)

        for r in resources:
            url = r.get("url", "")
            if url:
                if on_status:
                    on_status(f"MCP: URL recommandations trouvee")
                return url

    except Exception as e:
        if on_status:
            on_status(f"MCP indisponible ({e}) — fallback URL statique")

    return _FALLBACK_RECOMMENDATIONS_URL


# ── Download helpers ─────────────────────────────────────────────────────────


def _download(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


class _TextExtractor(HTMLParser):
    """Extract plain text from HTML."""
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        self._skip = tag in ("script", "style")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def html_to_text(html_content: str) -> str:
    """Convert HTML to plain text."""
    ext = _TextExtractor()
    ext.feed(html_content)
    return ext.get_text()


# ── Reports download ─────────────────────────────────────────────────────────


def download_reports_zip(year: int, url: str, on_status=None) -> list[dict]:
    """Download and extract reports for a given year.

    Returns list of {"filename": str, "text": str, "year": int, "size": int}.
    """
    cache_dir = DATA_DIR / f"rapports_{year}"
    index_file = cache_dir / "_index.json"

    if index_file.exists():
        if on_status:
            on_status(f"Rapports {year} (cache)")
        return json.loads(index_file.read_text(encoding="utf-8"))

    if on_status:
        on_status(f"Telechargement rapports {year}...")

    zip_data = _download(url)
    cache_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        html_files = [n for n in zf.namelist() if n.endswith(".html")]
        for name in html_files:
            content = zf.read(name).decode("utf-8", errors="replace")
            text = html_to_text(content)
            text = " ".join(text.split())
            if len(text) > 500:
                reports.append({
                    "filename": name.split("/")[-1],
                    "text": text,
                    "year": year,
                    "size": len(text),
                })

    index_file.write_text(json.dumps(reports, ensure_ascii=False), encoding="utf-8")

    if on_status:
        on_status(f"  {year}: {len(reports)} rapports extraits")

    return reports


def download_all_reports(years: list[int] | None = None, on_status=None) -> list[dict]:
    """Download reports for all requested years.

    Uses MCP to discover URLs dynamically, with fallback to static URLs.
    """
    report_zips = discover_report_zips(on_status=on_status)

    if years is not None:
        report_zips = {y: u for y, u in report_zips.items() if y in years}

    all_reports = []
    for year in sorted(report_zips):
        reports = download_reports_zip(year, report_zips[year], on_status=on_status)
        all_reports.extend(reports)

    all_reports.sort(key=lambda r: r["year"])
    return all_reports


# ── Recommendations download ────────────────────────────────────────────────


def download_recommendations(on_status=None) -> list[dict]:
    """Download the official recommendations file (2015-2018).

    Uses MCP to discover URL dynamically, with fallback to static URL.
    """
    cache_file = DATA_DIR / "recommandations_2015_2018.json"

    if cache_file.exists():
        if on_status:
            on_status("Recommandations (cache)")
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = discover_recommendations_url(on_status=on_status)

    if on_status:
        on_status("Telechargement recommandations 2015-2018...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = _download(url)
    text = data.decode("latin-1")
    lines = text.splitlines()

    recommendations = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3].strip():
            recommendations.append({
                "date": parts[0].strip(),
                "type": parts[1].strip(),
                "title": parts[2].strip(),
                "recommendation": parts[3].strip(),
                "destinataire_1": parts[4].strip() if len(parts) > 4 else "",
                "destinataire_2": parts[5].strip() if len(parts) > 5 else "",
                "destinataire_3": parts[6].strip() if len(parts) > 6 else "",
            })

    cache_file.write_text(json.dumps(recommendations, ensure_ascii=False), encoding="utf-8")

    if on_status:
        on_status(f"  {len(recommendations)} recommandations chargees")

    return recommendations


def clear_cache():
    """Remove all cached data."""
    import shutil
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR, ignore_errors=True)
