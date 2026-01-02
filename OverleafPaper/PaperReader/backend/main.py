"""
Paper Reader Backend - FastAPI Server
1980s Electric Pixel Style Paper Reader
"""
import os
import io
import re
import hashlib
import httpx
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
from PIL import Image
from pydantic import BaseModel
from fastapi import File, UploadFile, Body, WebSocket, WebSocketDisconnect
import json
import uuid
import random

from latex_compiler import LatexCompiler
from citation_mapper import CitationMapper

# Configuration
# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPERS_DIR = PROJECT_ROOT / "Papers"
THUMBNAILS_DIR = Path(__file__).parent / "thumbnails"
THUMBNAIL_WIDTH = 280
THUMBNAIL_HEIGHT = 400
PROJECTS_DIR = Path(__file__).parent / "projects"
PROJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,80}$")

# Create thumbnails directory
THUMBNAILS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Paper Reader API",
    description="1980s Electric Pixel Style Paper Reader Backend",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Compiler
latex_compiler = LatexCompiler(str(PROJECT_ROOT))

def resolve_project_root(project_id: Optional[str]) -> Path:
    if not project_id or project_id in {"server", "default"}:
        return PROJECT_ROOT

    project_id = project_id.strip()
    if not PROJECT_ID_PATTERN.match(project_id):
        raise HTTPException(status_code=400, detail="Invalid project_id")

    project_root = PROJECTS_DIR / project_id
    project_root.mkdir(parents=True, exist_ok=True)
    return project_root

# Presence Manager
class PresenceManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.user_info: dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str, info: dict):
        self.active_connections[client_id] = websocket
        self.user_info[client_id] = info
        await self.broadcast_presence()

    async def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.user_info:
            del self.user_info[client_id]
        await self.broadcast_presence()

    async def broadcast_presence(self):
        presence_data = list(self.user_info.values())
        message = json.dumps({
            "type": "presence",
            "users": presence_data,
            "count": len(presence_data)
        })
        
        # Create a list of tasks to run concurrently
        import asyncio
        disconnected = []
        for client_id, ws in self.active_connections.items():
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            await self.disconnect(client_id)

presence_manager = PresenceManager()

class CompileRequest(BaseModel):
    filename: str


def get_file_hash(filepath: Path) -> str:
    """Generate hash for file to use as cache key."""
    stat = filepath.stat()
    content = f"{filepath.name}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def build_llm_headers(request: Request) -> dict:
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


def generate_thumbnail(pdf_path: Path, force: bool = False) -> Path:
    """Generate thumbnail for PDF first page."""
    file_hash = get_file_hash(pdf_path)
    thumbnail_path = THUMBNAILS_DIR / f"{file_hash}.png"
    
    if thumbnail_path.exists() and not force:
        return thumbnail_path
    
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        # Calculate zoom to fit thumbnail size
        zoom_x = THUMBNAIL_WIDTH / page.rect.width
        zoom_y = THUMBNAIL_HEIGHT / page.rect.height
        zoom = min(zoom_x, zoom_y) * 2  # 2x for quality
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize to exact thumbnail size
        img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)
        
        # Create final image with padding
        final_img = Image.new("RGB", (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), (20, 20, 30))
        x = (THUMBNAIL_WIDTH - img.width) // 2
        y = (THUMBNAIL_HEIGHT - img.height) // 2
        final_img.paste(img, (x, y))
        
        final_img.save(thumbnail_path, "PNG", optimize=True)
        doc.close()
        
        return thumbnail_path
    except Exception as e:
        print(f"Error generating thumbnail for {pdf_path}: {e}")
        return None


def extract_paper_info(pdf_path: Path) -> dict:
    """Extract metadata from PDF."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = len(doc)
        file_size = pdf_path.stat().st_size
        doc.close()
        
        # Parse year and title from filename
        filename = pdf_path.stem
        parts = filename.split("_", 1)
        year = parts[0] if len(parts) > 1 and parts[0].isdigit() else ""
        title = parts[1] if len(parts) > 1 else filename
        
        return {
            "year": year,
            "title": title,
            "author": metadata.get("author", "Unknown"),
            "subject": metadata.get("subject", ""),
            "pages": page_count,
            "size_mb": round(file_size / (1024 * 1024), 2)
        }
    except Exception as e:
        return {
            "year": "",
            "title": pdf_path.stem,
            "author": "Unknown",
            "subject": "",
            "pages": 0,
            "size_mb": 0
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "paper-reader"}


@app.get("/api/papers")
async def list_papers(
    search: Optional[str] = Query(None, description="Search query"),
    sort_by: Optional[str] = Query("year", description="Sort by: year, title, size"),
    sort_order: Optional[str] = Query("desc", description="Sort order: asc, desc")
):
    """List all papers with metadata."""
    papers = []
    
    if not PAPERS_DIR.exists():
        return {"papers": [], "total": 0}
    
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        file_hash = get_file_hash(pdf_file)
        info = extract_paper_info(pdf_file)
        
        paper = {
            "id": file_hash,
            "filename": pdf_file.name,
            **info,
            "thumbnail_url": f"/api/thumbnail/{file_hash}",
            "pdf_url": f"/api/pdf/{file_hash}"
        }
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            if (search_lower not in paper["title"].lower() and 
                search_lower not in paper["year"] and
                search_lower not in paper.get("author", "").lower()):
                continue
        
        papers.append(paper)
    
    # Sort papers
    reverse = sort_order == "desc"
    if sort_by == "year":
        papers.sort(key=lambda x: x.get("year", "0"), reverse=reverse)
    elif sort_by == "title":
        papers.sort(key=lambda x: x.get("title", "").lower(), reverse=reverse)
    elif sort_by == "size":
        papers.sort(key=lambda x: x.get("size_mb", 0), reverse=reverse)
    
    return {"papers": papers, "total": len(papers)}


@app.get("/api/thumbnail/{paper_id}")
async def get_thumbnail(paper_id: str):
    """Get thumbnail for a paper."""
    # Find the PDF by hash
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        if get_file_hash(pdf_file) == paper_id:
            thumbnail_path = generate_thumbnail(pdf_file)
            if thumbnail_path and thumbnail_path.exists():
                return FileResponse(
                    thumbnail_path,
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"}
                )
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")


@app.get("/api/pdf/{paper_id}")
async def get_pdf(paper_id: str):
    """Get PDF file for viewing (inline, not download)."""
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        if get_file_hash(pdf_file) == paper_id:
            return FileResponse(
                pdf_file,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"inline; filename=\"{pdf_file.name}\"",
                    "Cache-Control": "public, max-age=3600"
                }
            )
    
    raise HTTPException(status_code=404, detail="PDF not found")


@app.get("/api/pdf/{paper_id}/download")
async def download_pdf(paper_id: str):
    """Download PDF file."""
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        if get_file_hash(pdf_file) == paper_id:
            return FileResponse(
                pdf_file,
                media_type="application/pdf",
                filename=pdf_file.name,
                headers={"Cache-Control": "public, max-age=3600"}
            )
    
    raise HTTPException(status_code=404, detail="PDF not found")


@app.get("/api/paper/{paper_id}/info")
async def get_paper_info(paper_id: str):
    """Get detailed info for a paper."""
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        if get_file_hash(pdf_file) == paper_id:
            info = extract_paper_info(pdf_file)
            return {
                "id": paper_id,
                "filename": pdf_file.name,
                **info
            }
    
    raise HTTPException(status_code=404, detail="Paper not found")


@app.post("/api/thumbnails/regenerate")
async def regenerate_thumbnails():
    """Regenerate all thumbnails."""
    count = 0
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        if generate_thumbnail(pdf_file, force=True):
            count += 1
    return {"regenerated": count}


# ==========================================
# OVERLEAF / COMPILER ENDPOINTS
# ==========================================

@app.get("/api/project/files")
async def list_project_files(project_id: Optional[str] = Query(None, description="Project ID")):
    """List .tex and .bib files in the project root."""
    project_root = resolve_project_root(project_id)
    files = []
    excludes = {".git", ".idea", "__pycache__", "venv", "node_modules"}
    
    for item in project_root.iterdir():
        if item.is_dir():
            continue
        if item.name.startswith(('.', '_')):
            continue
            
        if item.suffix in ['.tex', '.bib', '.cls', '.sty']:
            files.append({
                "name": item.name,
                "size": item.stat().st_size,
                "type": item.suffix[1:]
            })
            
    files.sort(key=lambda x: x["name"])
    return {"files": files}

@app.post("/api/compile")
async def compile_tex(request: CompileRequest, project_id: Optional[str] = Query(None, description="Project ID")):
    """Compile a tex file."""
    project_root = resolve_project_root(project_id)
    result = await latex_compiler.compile(request.filename, root_dir=project_root)
    return result

class SyncTexRequest(BaseModel):
    filename: str  # The PDF filename
    page: int
    x: float
    y: float

@app.post("/api/synctex")
async def synctex_to_source(request: SyncTexRequest, project_id: Optional[str] = Query(None, description="Project ID")):
    """
    Given PDF coordinates, return TeX source location.
    Uses 'synctex edit' command.
    """
    import subprocess
    import re
    
    # PDF is usually in PROJECT_ROOT
    project_root = resolve_project_root(project_id)
    pdf_path = project_root / request.filename
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
        
    # Command: synctex edit -o "PAGE:X:Y:PDF_FILE"
    # Note: synctex expects coordinates in points (1/72 inch)
    cmd = [
        "synctex", "edit", 
        "-o", f"{request.page}:{request.x}:{request.y}:{pdf_path}"
    ]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        output = proc.stdout
        
        input_match = re.search(r"Input:(.*)", output)
        line_match = re.search(r"Line:(\d+)", output)
        
        if input_match and line_match:
            source_file = Path(input_match.group(1).strip()).name
            line = int(line_match.group(1))
            return {
                "success": True,
                "file": source_file,
                "line": line
            }
        else:
            return {"success": False, "error": "No match found", "output": output}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

class ForwardSyncRequest(BaseModel):
    tex_file: str
    line: int
    pdf_file: str

@app.post("/api/synctex/forward")
async def synctex_to_pdf(request: ForwardSyncRequest, project_id: Optional[str] = Query(None, description="Project ID")):
    """
    Given TeX source location, return PDF coordinates.
    Uses 'synctex view' command.
    """
    import subprocess
    import re
    
    # Command: synctex view -i "LINE:COL:TEX_FILE" -o "PDF_FILE"
    cmd = [
        "synctex", "view",
        "-i", f"{request.line}:0:{request.tex_file}",
        "-o", f"{request.pdf_file}"
    ]
    
    try:
        project_root = resolve_project_root(project_id)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        output = proc.stdout
        
        page_match = re.search(r"Page:(\d+)", output)
        x_match = re.search(r"x:([\d\.]+)", output)
        y_match = re.search(r"y:([\d\.]+)", output)
        
        if page_match and x_match and y_match:
            return {
                "success": True,
                "page": int(page_match.group(1)),
                "x": float(x_match.group(1)),
                "y": float(y_match.group(1))
            }
        else:
            return {"success": False, "error": "No match found", "output": output}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/citations/map")
async def get_citation_map(project_id: Optional[str] = Query(None, description="Project ID")):
    """Parse ref.bib and map to existing papers."""
    project_root = resolve_project_root(project_id)
    bib_file = project_root / "ref.bib"
    
    papers = []
    for pdf_file in PAPERS_DIR.glob("*.pdf"):
        paper_id = get_file_hash(pdf_file)
        info = extract_paper_info(pdf_file)
        if info:
            info['id'] = paper_id
            papers.append(info)
    
    mapper = CitationMapper(papers)
    bib_entries = mapper.parse_bib_file(str(bib_file))
    mapping = mapper.map_citations(bib_entries)
    
    return {
        "mapping": mapping,
        "count": len(mapping),
        "total_refs": len(bib_entries)
    }

@app.get("/api/project/file/{filename}")
async def get_project_file(filename: str, project_id: Optional[str] = Query(None, description="Project ID")):
    """Serve a file from the project root (e.g. generated PDF)."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    project_root = resolve_project_root(project_id)
    file_path = project_root / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path)


@app.post("/api/llm/chat")
async def proxy_llm_chat(request: Request):
    payload = await request.json()
    headers = build_llm_headers(request)
    url = "https://game.agaii.org/llm/v1/chat/completions"

    try:
        if payload.get("stream"):
            async def stream():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url, json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            detail = await resp.aread()
                            raise HTTPException(status_code=resp.status_code, detail=detail.decode())
                        async for chunk in resp.aiter_bytes():
                            if chunk:
                                yield chunk

            return StreamingResponse(stream(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return JSONResponse(resp.json())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"LLM proxy error: {exc}") from exc


@app.get("/api/llm/models")
async def proxy_llm_models(request: Request):
    headers = build_llm_headers(request)
    url = "https://game.agaii.org/llm/v1/models"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return JSONResponse(resp.json())
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"LLM proxy error: {exc}") from exc

@app.get("/api/project/content/{filename}")
async def get_project_file_content(filename: str, project_id: Optional[str] = Query(None, description="Project ID")):
    """Read text content of a file."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    project_root = resolve_project_root(project_id)
    file_path = project_root / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        content = file_path.read_text(encoding='utf-8')
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SaveRequest(BaseModel):
    filename: str
    content: str

@app.post("/api/project/save")
async def save_project_file(request: SaveRequest, project_id: Optional[str] = Query(None, description="Project ID")):
    """Save text content to a file."""
    if ".." in request.filename or "/" in request.filename or "\\" in request.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    project_root = resolve_project_root(project_id)
    file_path = project_root / request.filename
    try:
        file_path.write_text(request.content, encoding='utf-8')
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/project/upload")
async def upload_project_file(project_id: Optional[str] = Query(None, description="Project ID"), file: UploadFile = File(...)):
    """Upload a file to PROJECT_ROOT."""
    filename = file.filename
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    project_root = resolve_project_root(project_id)
    file_path = project_root / filename
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"success": True, "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/project/file/{filename}")
async def delete_project_file(filename: str, project_id: Optional[str] = Query(None, description="Project ID")):
    """Delete a file from PROJECT_ROOT."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    project_root = resolve_project_root(project_id)
    file_path = project_root / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
        
    # Safety check: Don't delete compilation tools or main server
    if project_root == PROJECT_ROOT and filename in ["main.py", "latex_compiler.py", "citation_mapper.py", "requirements.txt"]:
        raise HTTPException(status_code=403, detail="Cannot delete system files")

    try:
        file_path.unlink()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.websocket("/ws/presence")
async def presence_websocket(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    try:
        # Wait for initial info message
        data = await websocket.receive_text()
        info = json.loads(data)
        if info.get("type") == "join":
            user_data = info.get("user", {})
            user_data["id"] = client_id
            await presence_manager.connect(websocket, client_id, user_data)
            
            while True:
                # Keep connection alive and listen for any updates
                try:
                    data = await websocket.receive_text()
                    # Could handle specific messages here (e.g. typing status)
                except WebSocketDisconnect:
                    break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await presence_manager.disconnect(client_id)


# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.responses import HTMLResponse
    
    @app.get("/")
    async def serve_index():
        """Serve the main index.html."""
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend not found")
    
    # Mount static files MUST BE LAST as it catches all sub-paths
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=22222)
