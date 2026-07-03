# Document Loading Services

The primary role of the loader is parsing diverse document formats like PDF, HTML, and Markdown, ensuring that relevant text along with accurate metadata is correctly parsed for downstream chunking.

When parsing HTML documents, metadata fields such as filename (source) and document type are preserved. Text scripts and styling nodes are stripped out.

For PDFs, PyMuPDF loops through every page, extracting text separately and mapping the page number in metadata as 'page'.
