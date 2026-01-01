import os
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("uvicorn")

class LatexCompiler:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.build_dir = self.root_dir / "PaperReader" / "backend" / "build"
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Attempt to auto-discover MiKTeX if not in PATH
        if shutil.which("pdflatex") is None:
            # Common paths for MiKTeX/TeXLive could be added here, currently just fixing for known user case
            known_paths = [
                r"D:\MikTeX\miktex\bin\x64",
                r"C:\Program Files\MiKTeX\miktex\bin\x64",
                r"C:\Program Files\MiKTeX 2.9\miktex\bin\x64",
            ]
            for p in known_paths:
                if os.path.exists(p) and (Path(p) / "pdflatex.exe").exists():
                    logger.info(f"Found non-PATH pdflatex at {p}, adding to PATH environment.")
                    os.environ["PATH"] += os.pathsep + p
                    break
        
    def check_pdflatex(self) -> bool:
        """Check if pdflatex is available in system PATH."""
        return shutil.which("pdflatex") is not None

    async def compile(self, tex_filename: str, root_dir: Optional[Path] = None) -> dict:
        """
        Compile a tex file using pdflatex -> bibtex -> pdflatex -> pdflatex sequence.
        Returns a dict with status and logs.
        """
        if not self.check_pdflatex():
            return {
                "success": False, 
                "log": "Error: pdflatex not found in system PATH. Please install TeX Live or MiKTeX."
            }

        compile_root = Path(root_dir) if root_dir else self.root_dir
        tex_file = compile_root / tex_filename
        if not tex_file.exists():
            return {"success": False, "log": f"Error: File {tex_filename} not found."}

        # Clean build directory slightly but keep previous run artifacts for speed if possible?
        # For now, let's just run in the root dir to avoid path issues with includes, 
        # but output to a build folder would be cleaner. 
        # However, LaTeX include paths are tricky. 
        # SAFEST APPROACH: Run in root_dir, let it generate auxiliary files there, 
        # effectively mimicking Overleaf.
        
        # We will capture stdout/stderr
        logs = []
        
        try:
            # 1. pdflatex
            logs.append(">> Running pdflatex (pass 1)...")
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-synctex=1", tex_filename],
                cwd=compile_root,
                capture_output=True,
                text=True
            )
            logs.append(proc.stdout)
            if proc.returncode != 0:
                logs.append(f">> pdflatex failed with code {proc.returncode}")
                # Don't return yet, sometimes it produces a PDF anyway
            
            # 2. bibtex (if aux exists)
            aux_file = tex_filename.replace(".tex", ".aux")
            if (compile_root / aux_file).exists():
                logs.append(">> Running bibtex...")
                proc = subprocess.run(
                    ["bibtex", aux_file.replace(".aux", "")],
                    cwd=compile_root,
                    capture_output=True,
                    text=True
                )
                logs.append(proc.stdout)
                # Bibtex failure might just mean no citations, continue
            
            # 3. pdflatex (pass 2)
            logs.append(">> Running pdflatex (pass 2)...")
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-synctex=1", tex_filename],
                cwd=compile_root,
                capture_output=True,
                text=True
            )
            logs.append(proc.stdout)
            
            # 4. pdflatex (pass 3)
            logs.append(">> Running pdflatex (pass 3)...")
            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-synctex=1", tex_filename],
                cwd=compile_root,
                capture_output=True,
                text=True
            )
            logs.append(proc.stdout)
            
            pdf_filename = tex_filename.replace(".tex", ".pdf")
            pdf_path = compile_root / pdf_filename
            
            if pdf_path.exists():
                return {
                    "success": True,
                    "log": "\n".join(logs),
                    "pdf_path": str(pdf_path)
                }
            else:
                return {
                    "success": False,
                    "log": "\n".join(logs) + "\n\nError: PDF file was not generated."
                }

        except Exception as e:
            logger.error(f"Compilation error: {e}")
            return {
                "success": False,
                "log": f"System Error during compilation: {str(e)}"
            }
