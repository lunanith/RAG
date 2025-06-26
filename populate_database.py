#!/usr/bin/env python3
"""
populate_database_confluence.py

Ingest all pages from a specified Confluence space,
chunk them into overlapping text segments,
store raw chunks in local PostgreSQL,
and index embeddings in a Chroma vector store.

Usage:
    python populate_database_confluence.py --space <SPACE_KEY> [--reset]

Options:
    --space <SPACE_KEY>    Confluence Space Key to ingest (required)
    --reset                Clear the local Chroma index before running
"""
import argparse
import os
import shutil
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from sqlalchemy import (
    create_engine, MetaData, Table,
    Column, String, Integer, Text, TIMESTAMP, text
)
from sqlalchemy.exc import IntegrityError

from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

# —————————————————————————————————————————————————————————
# Configuration
# —————————————————————————————————————————————————————————
# Confluence credentials via env vars
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")  # e.g. https://your-company.atlassian.net/wiki
CONFLUENCE_USER     = os.getenv("CONFLUENCE_USER")      # your email
CONFLUENCE_API_TOKEN= os.getenv("CONFLUENCE_API_TOKEN") # personal access token

# PostgreSQL (Homebrew) settings
DB_USER = os.getenv("DB_USER", os.getenv("USER"))
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "document_db")

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Chroma
CHROMA_PATH = "chroma"

# Create SQLAlchemy engine & table
engine = create_engine(DATABASE_URL, echo=False)
metadata = MetaData()
chunks_table = Table(
    "document_chunks", metadata,
    Column("id",          String(255), primary_key=True),
    Column("source",      Text,        nullable=False),
    Column("page",        Integer,     nullable=False),
    Column("chunk_index", Integer,     nullable=False),
    Column("content",     Text,        nullable=False),
    Column("inserted_at", TIMESTAMP(timezone=True), server_default=text("now()")),
)
metadata.create_all(engine)

# —————————————————————————————————————————————————————————
# Fetch Confluence pages
# —————————————————————————————————————————————————————————
AUTH = (CONFLUENCE_USER, CONFLUENCE_API_TOKEN)

def fetch_confluence_pages(space_key: str, limit: int = 50) -> list[Document]:
    """
    Retrieve all pages in the specified Confluence space.
    Returns a list of Documents (one per page).
    """
    docs = []
    start = 0
    base = CONFLUENCE_BASE_URL.rstrip('/')

    while True:
        resp = requests.get(
            f"{base}/rest/api/content",
            auth=AUTH,
            params={
                "spaceKey": space_key,
                "limit": limit,
                "start": start,
                "expand": "body.storage,version"
            }
        )
        resp.raise_for_status()
        data = resp.json()

        for page in data.get("results", []):
            page_id = page["id"]
            title   = page["title"]
            version = page.get("version", {}).get("number", 0)
            html    = page["body"]["storage"]["value"]

            # Strip HTML to text
            text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
            source_url = urljoin(base, f"/pages/{page_id}")

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": source_url,
                        "title": title,
                        "version": version
                    }
                )
            )

        # pagination check
        start += limit
        if start >= data.get("size", 0) + data.get("start", 0):
            break

    return docs

# —————————————————————————————————————————————————————————
# Chunking & ID Assignment
# —————————————————————————————————————————————————————————

def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
    last_key = None
    idx = 0
    for chunk in chunks:
        src  = chunk.metadata.get("source")
        page = chunk.metadata.get("title", "")  # use title or version as grouping
        key  = f"{src}"
        if key == last_key:
            idx += 1
        else:
            idx = 0
        chunk.metadata["id"] = f"{key}:{idx}"
        last_key = key
    return chunks

# —————————————————————————————————————————————————————————
# Persistence
# —————————————————————————————————————————————————————————

def store_chunks_to_postgres(chunks: list[Document]):
    with engine.begin() as conn:
        for chunk in chunks:
            m = chunk.metadata
            rec = {
                "id":          m["id"],
                "source":      m["source"],
                "page":        0,
                "chunk_index": int(m["id"].rsplit(":", 1)[1]),
                "content":     chunk.page_content,
            }
            try:
                conn.execute(chunks_table.insert().values(**rec))
            except IntegrityError:
                continue

# —————————————————————————————————————————————————————————
# Chroma Indexing
# —————————————————————————————————————————————————————————

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )
    existing_ids = set(db.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
    if new_chunks:
        print(f"👉 Adding {len(new_chunks)} new documents to Chroma")
        db.add_documents(new_chunks, ids=[c.metadata["id"] for c in new_chunks])
    else:
        print("✅ No new documents to add to Chroma")

# —————————————————————————————————————————————————————————
# Chroma Reset
# —————————————————————————————————————————————————————————

def clear_chroma():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

# —————————————————————————————————————————————————————————
# Main
# —————————————————————————————————————————————————————————

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--space", required=True,
        help="Confluence Space Key to ingest"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear the local Chroma index before ingestion"
    )
    args = parser.parse_args()

    if args.reset:
        clear_chroma()
        print("✨ Cleared Chroma index")

    print(f"🔍 Fetching Confluence space: {args.space}")
    documents = fetch_confluence_pages(args.space)

    print("✂️  Splitting documents into chunks")
    chunks = calculate_chunk_ids(split_documents(documents))

    print(f"💾 Storing {len(chunks)} chunks into Postgres")
    store_chunks_to_postgres(chunks)

    print("🔮 Indexing chunks into Chroma")
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# populate_database.py

# Scrape a webpage for PDF links (if --url provided) or load local PDFs from DATA_PATH,
# parse them into text chunks using LangChain,
# store chunks in a local PostgreSQL database,
# and index them into a Chroma vector store for retrieval.

# Usage:
#     python populate_database.py [--reset] [--url <webpage_url>]

# Options:
#     --reset            Clear the local Chroma database before running.
#     # --url <webpage_url>    If provided, scrape this page for PDF links instead of loading from DATA_PATH.
# """
# import argparse
# import os
# import shutil
# import tempfile
# import requests
# from urllib.parse import urljoin, urlparse
# from bs4 import BeautifulSoup

# from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from get_embedding_function import get_embedding_function
# from langchain_chroma import Chroma

# from sqlalchemy import (
#     create_engine, MetaData, Table,
#     Column, String, Integer, Text, TIMESTAMP, text
# )
# from sqlalchemy.exc import IntegrityError

# # —————————————————————————————————————————————————————————
# # Configuration
# # —————————————————————————————————————————————————————————
# CHROMA_PATH = "chroma"
# DATA_PATH   = "data"

# DB_USER = os.getenv("DB_USER", os.getenv("USER"))
# DB_PASS = os.getenv("DB_PASS", "")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "5432")
# DB_NAME = os.getenv("DB_NAME", "document_db")

# DATABASE_URL = (
#     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
#     f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# )
# engine = create_engine(DATABASE_URL, echo=False)

# metadata = MetaData()
# chunks_table = Table(
#     "document_chunks", metadata,
#     Column("id",          String(255), primary_key=True),
#     Column("source",      Text,        nullable=False),
#     Column("page",        Integer,     nullable=False),
#     Column("chunk_index", Integer,     nullable=False),
#     Column("content",     Text,        nullable=False),
#     Column("inserted_at", TIMESTAMP(timezone=True), server_default=text("now()")),
# )
# metadata.create_all(engine)


# def scrape_pdf_links(page_url: str) -> list[str]:
#     resp = requests.get(page_url)
#     resp.raise_for_status()
#     soup = BeautifulSoup(resp.text, "html.parser")
#     pdf_urls = set()
#     for a in soup.find_all("a", href=True):
#         href = a["href"].strip()
#         if href.lower().endswith(".pdf"):
#             pdf_urls.add(urljoin(page_url, href))
#     return list(pdf_urls)


# def load_documents_from_urls(urls: list[str]) -> list[Document]:
#     documents: list[Document] = []
#     for url in urls:
#         try:
#             # Download PDF to a temp file with correct Accept header
#             resp = requests.get(url, headers={"Accept": "application/pdf"}, stream=True)
#             resp.raise_for_status()
#             suffix = os.path.basename(urlparse(url).path) or "download.pdf"
#             fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(suffix)[1] or ".pdf")
#             os.close(fd)
#             with open(tmp_path, "wb") as f:
#                 for chunk in resp.iter_content(8192):
#                     f.write(chunk)
#             # Load pages from the local file
#             loader = PyPDFLoader(tmp_path)
#             pages = loader.load()
#             for doc in pages:
#                 doc.metadata["source"] = url
#             documents.extend(pages)
#         except Exception as e:
#             print(f"⚠️ Failed to fetch/load PDF {url}: {e}")
#         finally:
#             try:
#                 os.remove(tmp_path)
#             except Exception:
#                 pass
#     return documents


# def load_local_documents() -> list[Document]:
#     loader = PyPDFDirectoryLoader(DATA_PATH)
#     return loader.load()


# def split_documents(documents: list[Document]) -> list[Document]:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return splitter.split_documents(documents)


# def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
#     last_key = None
#     idx = 0
#     for chunk in chunks:
#         src = chunk.metadata.get("source")
#         page = chunk.metadata.get("page", 0)
#         key = f"{src}:{page}"
#         if key == last_key:
#             idx += 1
#         else:
#             idx = 0
#         chunk.metadata["id"] = f"{key}:{idx}"
#         last_key = key
#     return chunks


# def store_chunks_to_postgres(chunks: list[Document]):
#     with engine.begin() as conn:
#         for chunk in chunks:
#             m = chunk.metadata
#             rec = {
#                 "id":          m["id"],
#                 "source":      m["source"],
#                 "page":        int(m.get("page", 0)),
#                 "chunk_index": int(m["id"].rsplit(":", 1)[1]),
#                 "content":     chunk.page_content,
#             }
#             try:
#                 conn.execute(chunks_table.insert().values(**rec))
#             except IntegrityError:
#                 continue


# def add_to_chroma(chunks: list[Document]):
#     db = Chroma(
#         persist_directory=CHROMA_PATH,
#         embedding_function=get_embedding_function()
#     )
#     existing_ids = set(db.get(include=[])["ids"])
#     new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
#     if new_chunks:
#         print(f"👉 Adding {len(new_chunks)} new documents to Chroma")
#         ids = [c.metadata["id"] for c in new_chunks]
#         db.add_documents(new_chunks, ids=ids)
#     else:
#         print("✅ No new documents to add to Chroma")


# def clear_chroma():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--reset", action="store_true",
#         help="Clear the local Chroma database before running."
#     )
#     parser.add_argument(
#         "--url", type=str,
#         help="Webpage URL to scrape for PDF links instead of DATA_PATH"
#     )
#     args = parser.parse_args()

#     if args.reset:
#         clear_chroma()
#         print("✨ Cleared Chroma database")

#     if args.url:
#         print(f"🔍 Scraping PDFs from {args.url}")
#         pdf_urls = scrape_pdf_links(args.url)
#         print(f"Found {len(pdf_urls)} PDFs to process")
#         docs = load_documents_from_urls(pdf_urls)
#     else:
#         docs = load_local_documents()

#     print("✂️  Splitting documents into chunks")
#     chunks = split_documents(docs)
#     chunks = calculate_chunk_ids(chunks)

#     print(f"💾 Storing {len(chunks)} chunks into Postgres")
#     store_chunks_to_postgres(chunks)

#     print("🔮 Indexing chunks into Chroma")
#     add_to_chroma(chunks)

# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # populate_database.py

# # Scrape a webpage for PDF links (if --url provided) or load local PDFs from DATA_PATH,
# # parse them into text chunks using LangChain,
# # store chunks in a local PostgreSQL database,
# # and index them into a Chroma vector store for retrieval.

# # Usage:
# #     python populate_database.py [--reset] [--url <webpage_url>]

# # Options:
# #     --reset            Clear the local Chroma database before running.
# #     --url <webpage_url>    If provided, scrape this page for PDF links instead of loading from DATA_PATH.
# # """
# # import argparse
# # import os
# # import shutil
# # import requests
# # from urllib.parse import urljoin
# # from bs4 import BeautifulSoup


# # from langchain.document_loaders import PyPDFLoader
# # # from langchain_community.document_loaders import PyPDFDirectoryLoader, OnlinePDFLoader
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain.schema.document import Document
# # from get_embedding_function import get_embedding_function
# # from langchain_chroma import Chroma

# # from sqlalchemy import (
# #     create_engine, MetaData, Table,
# #     Column, String, Integer, Text, TIMESTAMP, text
# # )
# # from sqlalchemy.exc import IntegrityError

# # # —————————————————————————————————————————————————————————
# # # Configuration
# # # —————————————————————————————————————————————————————————
# # CHROMA_PATH = "chroma"
# # DATA_PATH   = "data"

# # DB_USER = os.getenv("DB_USER", os.getenv("USER"))
# # DB_PASS = os.getenv("DB_PASS", "")
# # DB_HOST = os.getenv("DB_HOST", "localhost")
# # DB_PORT = os.getenv("DB_PORT", "5432")
# # DB_NAME = os.getenv("DB_NAME", "document_db")

# # DATABASE_URL = (
# #     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
# #     f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# # )
# # engine = create_engine(DATABASE_URL, echo=False)

# # metadata = MetaData()
# # chunks_table = Table(
# #     "document_chunks", metadata,
# #     Column("id",          String(255), primary_key=True),
# #     Column("source",      Text,        nullable=False),
# #     Column("page",        Integer,     nullable=False),
# #     Column("chunk_index", Integer,     nullable=False),
# #     Column("content",     Text,        nullable=False),
# #     Column("inserted_at", TIMESTAMP(timezone=True), server_default=text("now()")),
# # )
# # # Ensure table exists
# # metadata.create_all(engine)


# # def scrape_pdf_links(page_url: str) -> list[str]:
# #     """
# #     Fetch the HTML at `page_url` and return all absolute .pdf URLs.
# #     """
# #     resp = requests.get(page_url)
# #     resp.raise_for_status()
# #     soup = BeautifulSoup(resp.text, "html.parser")

# #     pdf_urls = set()
# #     for a in soup.find_all("a", href=True):
# #         href = a["href"].strip()
# #         if href.lower().endswith(".pdf"):
# #             pdf_urls.add(urljoin(page_url, href))
# #     return list(pdf_urls)


# # # def load_documents_from_urls(urls: list[str]) -> list[Document]:
# # #     """
# # #     Download and parse each PDF URL into page-level Documents,
# # #     overwriting metadata['source'] with the original URL.
# # #     """
# # #     documents: list[Document] = []
# # #     for url in urls:
# # #         loader = OnlinePDFLoader(url)
# # #         pages = loader.load()
# # #         for doc in pages:
# # #             doc.metadata["source"] = url
# # #         documents.extend(pages)
# # #     return documents

# # def load_documents_from_urls(urls: list[str]) -> list[Document]:
# #     documents: list[Document] = []
# #     for url in urls:
# #         loader = PyPDFLoader(url)
# #         pages = loader.load()           # one Document per page
# #         # Overwrite the source so you see the URL instead of a temp-file path
# #         for doc in pages:
# #             doc.metadata["source"] = url
# #         documents.extend(pages)
# #     return documents



# # def load_local_documents() -> list[Document]:
# #     """
# #     Load all PDFs from the local DATA_PATH directory.
# #     """
# #     loader = PyPDFDirectoryLoader(DATA_PATH)
# #     return loader.load()


# # def split_documents(documents: list[Document]) -> list[Document]:
# #     splitter = RecursiveCharacterTextSplitter(
# #         chunk_size=800,
# #         chunk_overlap=80,
# #         length_function=len,
# #         is_separator_regex=False,
# #     )
# #     return splitter.split_documents(documents)


# # def calculate_chunk_ids(chunks: list[Document]) -> list[Document]:
# #     """
# #     Assign deterministic IDs `"<source>:<page>:<chunk_index>"` to each chunk.
# #     """
# #     last_key = None
# #     idx = 0
# #     for chunk in chunks:
# #         src = chunk.metadata.get("source")
# #         page = chunk.metadata.get("page", 0)
# #         key = f"{src}:{page}"
# #         if key == last_key:
# #             idx += 1
# #         else:
# #             idx = 0
# #         chunk.metadata["id"] = f"{key}:{idx}"
# #         last_key = key
# #     return chunks


# # def store_chunks_to_postgres(chunks: list[Document]):
# #     """
# #     Insert each chunk into `document_chunks`, skipping any IDs that already exist.
# #     """
# #     with engine.begin() as conn:
# #         for chunk in chunks:
# #             m = chunk.metadata
# #             rec = {
# #                 "id":          m["id"],
# #                 "source":      m["source"],
# #                 "page":        int(m.get("page", 0)),
# #                 "chunk_index": int(m["id"].rsplit(":", 1)[1]),
# #                 "content":     chunk.page_content,
# #             }
# #             try:
# #                 conn.execute(chunks_table.insert().values(**rec))
# #             except IntegrityError:
# #                 # already exists
# #                 continue


# # def add_to_chroma(chunks: list[Document]):
# #     db = Chroma(
# #         persist_directory=CHROMA_PATH,
# #         embedding_function=get_embedding_function()
# #     )
# #     existing_ids = set(db.get(include=[])["ids"])
# #     new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
# #     if new_chunks:
# #         print(f"👉 Adding {len(new_chunks)} new documents to Chroma")
# #         ids = [c.metadata["id"] for c in new_chunks]
# #         db.add_documents(new_chunks, ids=ids)
# #     else:
# #         print("✅ No new documents to add to Chroma")


# # def clear_chroma():
# #     if os.path.exists(CHROMA_PATH):
# #         shutil.rmtree(CHROMA_PATH)


# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument(
# #         "--reset", action="store_true",
# #         help="Clear the local Chroma database before running."
# #     )
# #     parser.add_argument(
# #         "--url", type=str,
# #         help="Webpage URL to scrape for PDF links instead of DATA_PATH"
# #     )
# #     args = parser.parse_args()

# #     if args.reset:
# #         clear_chroma()
# #         print("✨ Cleared Chroma database")

# #     if args.url:
# #         print(f"🔍 Scraping PDFs from {args.url}")
# #         pdf_urls = scrape_pdf_links(args.url)
# #         print(f"Found {len(pdf_urls)} PDFs to process")
# #         docs = load_documents_from_urls(pdf_urls)
# #     else:
# #         docs = load_local_documents()

# #     print("✂️  Splitting documents into chunks")
# #     chunks = split_documents(docs)
# #     chunks = calculate_chunk_ids(chunks)

# #     print(f"💾 Storing {len(chunks)} chunks into Postgres")
# #     store_chunks_to_postgres(chunks)

# #     print("🔮 Indexing chunks into Chroma")
# #     add_to_chroma(chunks)

# # if __name__ == "__main__":
# #     main()

# # # #!/usr/bin/env python3
# # # import argparse
# # # import os
# # # import shutil

# # # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # # from langchain.schema.document import Document
# # # from get_embedding_function import get_embedding_function
# # # from langchain_chroma import Chroma

# # # from sqlalchemy import (
# # #     create_engine, MetaData, Table, Column,
# # #     String, Integer, Text, TIMESTAMP, text
# # # )
# # # from sqlalchemy.exc import IntegrityError

# # # # —————————————————————————————————————————————————————————
# # # # 1) PostgreSQL setup (local via Homebrew)
# # # # —————————————————————————————————————————————————————————

# # # # You can set these in your shell; defaults assume macOS user with no password:
# # # DB_USER = os.getenv("DB_USER", os.getenv("USER"))
# # # DB_PASS = os.getenv("DB_PASS", "")
# # # DB_HOST = os.getenv("DB_HOST", "localhost")
# # # DB_PORT = os.getenv("DB_PORT", "5432")
# # # DB_NAME = os.getenv("DB_NAME", "document_db")

# # # DATABASE_URL = (
# # #     f"postgresql+psycopg2://{DB_USER}:{DB_PASS}"
# # #     f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# # # )
# # # engine = create_engine(DATABASE_URL, echo=False)

# # # metadata = MetaData()
# # # chunks_table = Table(
# # #     "document_chunks",
# # #     metadata,
# # #     Column("id",          String(255),   primary_key=True),
# # #     Column("source",      Text,          nullable=False),
# # #     Column("page",        Integer,       nullable=False),
# # #     Column("chunk_index", Integer,       nullable=False),
# # #     Column("content",     Text,          nullable=False),
# # #     Column("inserted_at", TIMESTAMP(timezone=True),
# # #            server_default=text("now()")),
# # # )
# # # # Create table if it doesn't exist
# # # metadata.create_all(engine)


# # # def store_chunks_to_postgres(chunks: list[Document]):
# # #     """
# # #     Insert each chunk into `document_chunks`
# # #     unless its `id` already exists.
# # #     """
# # #     with engine.begin() as conn:
# # #         for chunk in chunks:
# # #             meta = chunk.metadata
# # #             # id looks like "data/foo.pdf:3:1"
# # #             rec = {
# # #                 "id":          meta["id"],
# # #                 "source":      meta["source"],
# # #                 "page":        int(meta["page"]),
# # #                 "chunk_index": int(meta["id"].rsplit(":", 1)[1]),
# # #                 "content":     chunk.page_content,
# # #             }
# # #             try:
# # #                 conn.execute(chunks_table.insert().values(**rec))
# # #             except IntegrityError:
# # #                 # already in DB
# # #                 pass


# # # # —————————————————————————————————————————————————————————
# # # # 2) Original pipeline code
# # # # —————————————————————————————————————————————————————————

# # # CHROMA_PATH = "chroma"
# # # DATA_PATH = "data"


# # # def load_documents():
# # #     loader = PyPDFDirectoryLoader(DATA_PATH)
# # #     return loader.load()


# # # def split_documents(documents: list[Document]):
# # #     splitter = RecursiveCharacterTextSplitter(
# # #         chunk_size=800,
# # #         chunk_overlap=80,
# # #         length_function=len,
# # #         is_separator_regex=False,
# # #     )
# # #     return splitter.split_documents(documents)


# # # def calculate_chunk_ids(chunks: list[Document]):
# # #     last_page_id = None
# # #     current_chunk_index = 0
# # #     for chunk in chunks:
# # #         source = chunk.metadata.get("source")
# # #         page   = chunk.metadata.get("page")
# # #         current_page_id = f"{source}:{page}"
# # #         if current_page_id == last_page_id:
# # #             current_chunk_index += 1
# # #         else:
# # #             current_chunk_index = 0
# # #         chunk_id = f"{current_page_id}:{current_chunk_index}"
# # #         last_page_id = current_page_id
# # #         chunk.metadata["id"] = chunk_id
# # #     return chunks


# # # def add_to_chroma(chunks: list[Document]):
# # #     db = Chroma(
# # #         persist_directory=CHROMA_PATH,
# # #         embedding_function=get_embedding_function()
# # #     )
# # #     # fetch existing IDs
# # #     existing = db.get(include=[])
# # #     existing_ids = set(existing["ids"])
# # #     print(f"Number of existing documents in Chroma: {len(existing_ids)}")

# # #     # only add new
# # #     new_chunks = [c for c in chunks if c.metadata["id"] not in existing_ids]
# # #     if new_chunks:
# # #         print(f"👉 Adding {len(new_chunks)} new documents to Chroma")
# # #         ids = [c.metadata["id"] for c in new_chunks]
# # #         db.add_documents(new_chunks, ids=ids)
# # #     else:
# # #         print("✅ No new documents to add to Chroma")


# # # def clear_chroma():
# # #     if os.path.exists(CHROMA_PATH):
# # #         shutil.rmtree(CHROMA_PATH)


# # # # —————————————————————————————————————————————————————————
# # # # 3) Glue it together in main()
# # # # —————————————————————————————————————————————————————————

# # # def main():
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument(
# # #         "--reset", action="store_true",
# # #         help="Clear the local Chroma database before running."
# # #     )
# # #     args = parser.parse_args()

# # #     if args.reset:
# # #         print("✨ Clearing Chroma database")
# # #         clear_chroma()

# # #     # Load, split, assign IDs
# # #     docs   = load_documents()
# # #     chunks = split_documents(docs)
# # #     chunks = calculate_chunk_ids(chunks)

# # #     # Persist to Postgres
# # #     print(f"Storing {len(chunks)} chunks into Postgres…")
# # #     store_chunks_to_postgres(chunks)

# # #     # (Optional) also update Chroma
# # #     add_to_chroma(chunks)


# # # if __name__ == "__main__":
# # #     main()

