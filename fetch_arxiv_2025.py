
import urllib.request
import feedparser
import time
import os

# Base API query URL
base_url = 'http://export.arxiv.org/api/query?'

# Search query
# Expanded Search query
# "Knowledge Graph Question Answering", KGQA, KBQA, "Knowledge Base Question Answering", "Text-to-SPARQL", "Reasoning on Knowledge Graphs"
search_query = 'all:("Knowledge Graph Question Answering" OR KGQA OR "Question Answering over Knowledge Graphs" OR "Question Answering on Knowledge Graphs" OR "Knowledge Base Question Answering" OR KBQA OR "Text-to-SPARQL" OR "Complex Question Answering" OR "Reasoning on Knowledge Graphs")'

target_count = 120 # Aiming a bit higher to ensure we have enough valid ones
folder_path = "/data/Yanlai/KGLLM/references/"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Count existing files
existing_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
current_count = len(existing_files)
print(f"Current paper count: {current_count}")

start_index = 0
max_results_per_page = 100

while current_count < target_count:
    print(f"Fetching results starting at {start_index}...")
    
    params = {
        'search_query': search_query,
        'start': start_index,
        'max_results': max_results_per_page,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    query_string = urllib.parse.urlencode(params)
    url = base_url + query_string

    try:
        response = urllib.request.urlopen(url)
        feed = feedparser.parse(response)
    except Exception as e:
        print(f"Error fetching feed: {e}")
        time.sleep(5)
        continue

    if not feed.entries:
        print("No more entries found.")
        break

    fetched_in_batch = 0
    for entry in feed.entries:
        if current_count >= target_count:
            break

        published = entry.published
        year = published[:4]
        
        # We only want 2025 papers
        if year != '2025':
            continue

        safe_title = "".join([c if c.isalnum() else "_" for c in entry.title])
        safe_title = safe_title[:150]
        filename = f"{folder_path}{safe_title}.pdf"

        if os.path.exists(filename):
            # print(f"Skipping {entry.title} (already exists)")
            continue

        print(f"Found new paper: {entry.title}")
        pdf_link = entry.id.replace('abs', 'pdf')
        
        # Download
        try:
            download_url = pdf_link
            if not download_url.endswith('.pdf'):
                download_url += ".pdf"
            
            req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
            
            print(f"Downloaded: {filename}")
            current_count += 1
            fetched_in_batch += 1
            time.sleep(1)
        except Exception as e:
            print(f"Failed to download {pdf_link}: {e}")

    print(f"Batch finished. Downloaded {fetched_in_batch} new papers in this batch.")
    
    # If we didn't find any 2025 papers in this batch, and we are sorting by date descending,
    # it might mean we passed 2025.
    # Check the last entry's date.
    if feed.entries:
        last_entry_year = feed.entries[-1].published[:4]
        if last_entry_year < '2025':
            print("Reached papers older than 2025. Stopping.")
            break
    
    start_index += max_results_per_page
    time.sleep(2)

print(f"Final paper count: {current_count}")
