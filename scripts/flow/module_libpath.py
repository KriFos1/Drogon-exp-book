"""Print LD_LIBRARY_PATH of an Lmod module and its dependencies (from the .lua files)."""
import re, sys, os
seen, order = set(), []
def visit(name):
    if name in seen:
        return
    seen.add(name)
    path = f"/cluster/modulefiles/all/{name}.lua"
    text = open(path).read()
    for dep in re.findall(r'load\("([^"]+)"\)', text):
        visit(dep)
    root = re.search(r'local root = "([^"]+)"', text)
    for rel in re.findall(r'prepend_path\("LD_LIBRARY_PATH", pathJoin\(root, "([^"]+)"\)\)', text):
        order.append(os.path.join(root.group(1), rel))
visit(sys.argv[1])
print(":".join(dict.fromkeys(reversed(order))))
