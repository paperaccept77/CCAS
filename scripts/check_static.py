"""Check source syntax, module imports and public fixture shapes."""
import ast
import json
from pathlib import Path


def check_local_imports(root):
    modules = {}
    for path in (root / "src/cath").rglob("*.py"):
        parts = list(path.relative_to(root / "src").with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if is_package else parts)
        modules[name] = (ast.parse(path.read_text(encoding="utf-8")), is_package)
    for name, (tree, is_package) in modules.items():
        package = name.split(".") if is_package else name.split(".")[:-1]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            base = package[:len(package)-node.level+1]
            target = ".".join(base + (node.module.split(".") if node.module else []))
            assert target in modules, f"{name}: missing import module {target}"
            exports = set()
            for item in modules[target][0].body:
                if isinstance(item, (ast.FunctionDef, ast.ClassDef)):
                    exports.add(item.name)
                elif isinstance(item, ast.ImportFrom):
                    exports.update(alias.asname or alias.name for alias in item.names)
                elif isinstance(item, ast.Assign):
                    exports.update(t.id for t in item.targets if isinstance(t, ast.Name))
            for alias in node.names:
                assert alias.name in exports or f"{target}.{alias.name}" in modules, (
                    f"{name}: missing symbol {target}.{alias.name}")
    return len(modules)


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = check_local_imports(root)
    config = json.loads((root / "configs/paper.json").read_text(encoding="utf-8"))
    json_files = list((root / "configs").glob("*.json")) + list((root / "examples").glob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for path in (root / "examples").glob("*.jsonl"):
        records[path.name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for request in records["requests.jsonl"]:
        assert len(request["request_features"]) == config["dim"]
        assert len(request["base_scores"]) == len(request["candidates"])
        assert all(visit["timestamp"] < request["cutoff"] for visit in request["history"])
        assert "positive_poi_id" not in request
        for poi in request["candidates"] + [visit["poi"] for visit in request["history"]]:
            assert poi["mask"][0] == (poi["text"] is not None)
            assert poi["mask"][1] == (poi["image_embedding"] is not None)
            if poi["image_embedding"] is not None:
                assert len(poi["image_embedding"]) == config["image_dim"]
            if "features" in poi:
                assert len(poi["features"]) == config["dim"]
    for row in records["index.jsonl"]:
        assert row["split"] == "train" and len(row["embedding"]) == config["dim"]
    # Verify all cost-bearing declared operations have an explicit upper bound.
    module = ast.parse((root / "src/cath/types.py").read_text(encoding="utf-8"))
    operations = next(node.value for node in module.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "OPERATIONS" for target in node.targets))
    costs = json.loads((root / "examples/costs.example.json").read_text(encoding="utf-8"))
    for role, values in zip(operations.keys, operations.values):
        for value in values.elts:
            if value.value not in {"finish", "request_peer"}:
                assert f"{role.attr.lower()}.{value.value}" in costs["operation_upper"]
    print(f"Static checks passed: {len(files)} Python files, {modules} modules, {len(json_files)} JSON files, {sum(map(len, records.values()))} JSONL records.")


if __name__ == "__main__":
    main()
