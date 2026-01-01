import re
import difflib
from pathlib import Path
import logging

logger = logging.getLogger("uvicorn")

class CitationMapper:
    def __init__(self, papers_metadata: list):
        """
        papers_metadata: list of dicts from main.py, e.g.
        [{'id': '...', 'title': '...', 'filename': '...'}, ...]
        """
        self.papers = papers_metadata
        
    def parse_bib_file(self, bib_path: str) -> dict:
        """
        Simple regex-based bib parser.
        Returns dict: {citation_key: title}
        """
        bib_path = Path(bib_path)
        if not bib_path.exists():
            return {}

        try:
            content = bib_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to read bib file: {e}")
            return {}

        # Regex to find @type{key, ... title={...} ... }
        # This is a basic parser and might not handle all edge cases (nested braces etc)
        # but sufficient for standard generated bibs.
        
        entries = {}
        
        # Regex explanation:
        # @\w+\s*{\s*([^,]+),      --> matches @type{key,
        # (?:[^{}]|{[^{}]*})*      --> matches content before title (non-greedy)
        # \btitle\s*=\s*{([^}]+)}  --> matches title={Title Content}
        
        # Simpler approach: split by @, then parse key and title line manually
        raw_entries = content.split('@')
        for entry in raw_entries:
            if not entry.strip(): 
                continue
                
            lines = entry.split('\n')
            first_line = lines[0].strip()
            
            # Extract key: type{key,
            match_key = re.match(r'^\w+\s*{\s*([^,]+),', first_line)
            if not match_key:
                continue
            
            key = match_key.group(1).strip()
            
            # Extract title loop
            title = None
            for line in lines:
                line = line.strip()
                # Check for title = {Some Title}, or title="Some Title"
                # Handle simplified cases common in academic bibs
                if line.lower().startswith('title'):
                    # Remove 'title', '=', '{', '}', '"', ','
                    clean_line = re.sub(r'^title\s*=\s*[{"\']?', '', line, flags=re.IGNORECASE)
                    clean_line = re.sub(r'[}"\'],?$', '', clean_line)
                    title = clean_line
                    break
            
            if key and title:
                entries[key] = title
                
        return entries

    def map_citations(self, bib_entries: dict) -> dict:
        """
        Maps citation keys to paper IDs using fuzzy matching on titles.
        Returns: {citation_key: paper_id}
        """
        mapping = {}
        
        # Create a list of normalized titles from papers for fuzzy matching
        paper_titles = [p['title'].lower() for p in self.papers]
        
        for key, title in bib_entries.items():
            if not title:
                continue
                
            norm_title = title.lower()
            
            # 1. Exact match attempt (fast)
            # 2. Fuzzy match
            matches = difflib.get_close_matches(norm_title, paper_titles, n=1, cutoff=0.7)
            
            if matches:
                matched_title = matches[0]
                # Find the paper ID for this title
                # (Assuming titles are unique enough or taking the first one)
                for p in self.papers:
                    if p['title'].lower() == matched_title:
                        mapping[key] = p['id']
                        break
        
        return mapping
