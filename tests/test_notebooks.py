import json
from pathlib import Path


def test_four_notebooks_parse():
    notebooks = sorted(Path("notebooks").glob("*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        data = json.loads(path.read_text())
        assert data["nbformat"] == 4
        assert any(cell["cell_type"] == "code" for cell in data["cells"])
