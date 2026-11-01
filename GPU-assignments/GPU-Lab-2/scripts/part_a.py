"""HW4.5 Part A: HW4 fine tune unchanged, batch doubled until OOM."""
from common import *

tr, _, _, _, names, info = load_data()
rows = probe(tr, len(names))
save("part_a", dict(data=info, rows=rows, max_batch=max(r["batch"] for r in rows if not r["oom"])))
