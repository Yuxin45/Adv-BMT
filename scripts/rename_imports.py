# tools/rename_imports.py
import ast, pathlib, sys

OLD, NEW = "infgen", "bmt"

def rewrite_file(path: pathlib.Path):
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return 0
    lines = src.splitlines(keepends=True)
    changed = False

    for node in ast.walk(tree):
        # from infgen... import ...
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == OLD or node.module.startswith(OLD + "."):
                new_mod = NEW + node.module[len(OLD):]
                # ast gives line/col;直接替换该行的模块名片段
                line = lines[node.lineno - 1]
                prefix = line[:node.col_offset]
                rest = line[node.col_offset:]
                rest = rest.replace(f"from {node.module} import", f"from {new_mod} import", 1)
                lines[node.lineno - 1] = prefix + rest
                changed = True

        # import infgen[.sub]
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == OLD or name.startswith(OLD + "."):
                    new_name = NEW + name[len(OLD):]
                    line = lines[node.lineno - 1]
                    prefix = line[:node.col_offset]
                    rest = line[node.col_offset:]
                    rest = rest.replace(name, new_name, 1)
                    lines[node.lineno - 1] = prefix + rest
                    changed = True

    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return int(changed)

def main(root):
    n = 0
    for p in pathlib.Path(root).rglob("*.py"):
        # 跳过虚拟环境/构建目录
        if any(seg in p.parts for seg in (".venv", "venv", ".git", ".tox", "build", "dist", "__pycache__")):
            continue
        n += rewrite_file(p)
    print(f"Rewritten files: {n}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
