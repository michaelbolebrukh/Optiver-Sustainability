"""Execute and save the notebook, preserving completed output if a cell fails."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('carbon_portfolio.ipynb')
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(Path.cwd())}})
client.on_cell_start = lambda cell, cell_index, **kwargs: print(f'Cell {cell_index}', flush=True)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)
print('Notebook executed successfully.', flush=True)
