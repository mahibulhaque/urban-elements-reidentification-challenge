"""Build a self-contained HTML viewer for manual inspection of validation
re-ranking results.

For each val query: shows the query thumbnail + top-K gallery images sorted
by per-class re-rank distance with full metadata (pid, camid, class, sim,
match flag).

Reads the cached val features from feat_cache/<tag>_val/ and the per-class
best params from submissions/eval_<tag>.json (so the displayed ranking is
exactly the one used in the submission for this tag).
"""
import argparse, csv, json, os, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.re_ranking import re_ranking  # noqa: E402

DATA = os.path.join(REPO, 'Dataset/UrbanUAM_Merged')


def norm_cls(c):
    c = c.strip().lower()
    return 'trafficsign' if c == 'trafficsignal' else c


def read_classes(p):
    d = {}
    with open(p, newline='') as f:
        rd = csv.reader(f); hdr = next(rd)
        ni = hdr.index('imageName'); ci = hdr.index('Class')
        for r in rd: d[r[ni]] = norm_cls(r[ci])
    return d


def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='Hplus',
                    help='feature tag (e.g. Hplus). Reads feat_cache/<tag>_val and submissions/eval_<tag>.json')
    ap.add_argument('--top_k', type=int, default=30,
                    help='top-K gallery per query to embed (keep small to keep HTML reasonable)')
    ap.add_argument('--out', default='val_viewer.html')
    args = ap.parse_args()

    feat_dir = os.path.join(REPO, 'feat_cache', f'{args.tag}_val')
    eval_json = os.path.join(REPO, 'submissions', f'eval_{args.tag}.json')

    qf = l2(np.load(os.path.join(feat_dir, 'qf.npy'))).astype(np.float32)
    gf = l2(np.load(os.path.join(feat_dir, 'gf.npy'))).astype(np.float32)
    with open(os.path.join(feat_dir, 'meta.json')) as f:
        meta = json.load(f)
    with open(eval_json) as f:
        ev = json.load(f)
    pc = ev['perclass_params']

    q_pid = np.asarray(meta['query_pids'])
    g_pid = np.asarray(meta['gallery_pids'])
    q_cam = np.asarray(meta['query_camids'])
    g_cam = np.asarray(meta['gallery_camids'])

    # Class labels for val (separate q/g maps to avoid basename collisions)
    nq = read_classes(os.path.join(DATA, 'val_query_classes.csv'))
    ng = read_classes(os.path.join(DATA, 'val_test_classes.csv'))
    q_cls = np.array([nq[os.path.basename(p)] for p in meta['query_paths']])
    g_cls = np.array([ng[os.path.basename(p)] for p in meta['gallery_paths']])

    # Per-class re-rank distance, take top-K within class for each query.
    # For queries we keep one record with the ranking + metadata.
    records = []
    print(f'computing per-class rankings for {args.tag} ...')
    for cls in np.unique(q_cls):
        qi = np.where(q_cls == cls)[0]
        gi_same = np.where(g_cls == cls)[0]
        if len(qi) == 0 or len(gi_same) == 0:
            continue
        params = pc[cls]['params']
        k1, k2, lam = params
        qf_c, gf_c = qf[qi], gf[gi_same]
        dist = re_ranking(qf_c @ gf_c.T, qf_c @ qf_c.T, gf_c @ gf_c.T,
                          k1=k1, k2=k2, lambda_value=lam)
        # Also pure cosine sim for reporting
        sim = qf_c @ gf_c.T
        for li, gq in enumerate(qi):
            order = np.argsort(dist[li])[:args.top_k]
            top_global = gi_same[order]
            ap_match_count = 0
            ranked = []
            for r, gidx in enumerate(top_global):
                is_match = bool(int(g_pid[gidx]) == int(q_pid[gq]) and int(g_cam[gidx]) != int(q_cam[gq]))
                if is_match: ap_match_count += 1
                ranked.append({
                    'rank': r + 1,
                    'path': os.path.relpath(meta['gallery_paths'][gidx], REPO),
                    'pid':  int(g_pid[gidx]),
                    'cam':  int(g_cam[gidx]),
                    'cls':  str(g_cls[gidx]),
                    'rerank_dist': float(dist[li, order[r]]),
                    'cos_sim': float(sim[li, order[r]]),
                    'match': is_match,
                })
            records.append({
                'qidx': int(gq),
                'qpath': os.path.relpath(meta['query_paths'][gq], REPO),
                'qpid':  int(q_pid[gq]),
                'qcam':  int(q_cam[gq]),
                'qcls':  str(q_cls[gq]),
                'params_used': {'k1': int(k1), 'k2': int(k2), 'lam': float(lam)},
                'top': ranked,
                'n_match_in_topK': ap_match_count,
                'n_total_match':   int(((g_pid == int(q_pid[gq])) & (g_cls == cls) &
                                        (g_cam != int(q_cam[gq]))).sum()),
            })

    # Sort by qidx so they're in 1..n order for human browsing
    records.sort(key=lambda r: r['qidx'])
    print(f'collected {len(records)} queries  (top_k={args.top_k} each)')

    # ---------- HTML template ----------
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Val Viewer — TAG_PLACEHOLDER</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font: 13px/1.4 -apple-system, system-ui, sans-serif; color: #222; background: #f7f7f7; display: flex; height: 100vh; }
  #sidebar { width: 280px; height: 100vh; overflow-y: auto; background: #fff; border-right: 1px solid #ddd; padding: 8px; }
  #sidebar h3 { margin: 0 0 6px; font-size: 12px; color: #888; text-transform: uppercase; }
  #search { width: 100%; padding: 6px; margin-bottom: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
  .qrow { display: flex; align-items: center; gap: 6px; padding: 4px 6px; cursor: pointer; border-radius: 4px; margin: 1px 0; }
  .qrow:hover { background: #eef; }
  .qrow.active { background: #cce; font-weight: 600; }
  .qrow img { width: 38px; height: 76px; object-fit: cover; border-radius: 3px; background: #ddd; }
  .qrow .meta { font-size: 11px; color: #555; line-height: 1.2; }
  .qrow .meta strong { color: #222; }
  #main { flex: 1; padding: 16px; overflow-y: auto; }
  #qheader { display: flex; gap: 16px; align-items: flex-start; padding-bottom: 12px; border-bottom: 1px solid #ddd; margin-bottom: 12px; }
  #qheader img { height: 220px; border-radius: 6px; border: 2px solid #444; }
  #qheader .info { font-size: 14px; }
  #qheader .info b { display: inline-block; min-width: 90px; color: #555; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
  .card { background: #fff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.1); overflow: hidden; }
  .card.match { outline: 3px solid #2ecc71; }
  .card.nomatch { outline: 3px solid #e74c3c; }
  .card img { width: 100%; height: 280px; object-fit: cover; display: block; background: #ddd; }
  .card .caption { padding: 6px 8px; font-size: 11px; line-height: 1.3; }
  .card .caption .rank { font-size: 14px; font-weight: 600; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 10px; color: #fff; margin-left: 4px; }
  .badge.match { background: #2ecc71; }
  .badge.nomatch { background: #e74c3c; }
  .empty { color: #888; text-align: center; padding: 60px; font-size: 16px; }
</style>
</head><body>
<div id="sidebar">
  <h3>Queries (TAG_PLACEHOLDER)</h3>
  <input id="search" placeholder="filter: id / cam / class">
  <div id="thumbs"></div>
</div>
<div id="main">
  <div id="qheader" class="empty">Pick a query from the left.</div>
  <div id="grid"></div>
</div>
<script>
const DATA = DATA_PLACEHOLDER;
const $ = (id) => document.getElementById(id);

function renderSidebar(filter='') {
  const t = $('thumbs'); t.innerHTML = '';
  const f = filter.toLowerCase();
  for (const r of DATA) {
    if (f) {
      const blob = `${r.qpid} c${r.qcam} ${r.qcls}`.toLowerCase();
      if (!blob.includes(f)) continue;
    }
    const div = document.createElement('div');
    div.className = 'qrow'; div.dataset.qidx = r.qidx;
    div.innerHTML = `
      <img src="${r.qpath}">
      <div class="meta">
        <strong>id ${r.qpid}</strong><br>
        c${r.qcam}<br>${r.qcls}<br>${r.n_match_in_topK}/${r.n_total_match} hits
      </div>`;
    div.onclick = () => selectQuery(r.qidx);
    t.appendChild(div);
  }
}

function selectQuery(qidx) {
  document.querySelectorAll('.qrow').forEach(e => e.classList.toggle('active', +e.dataset.qidx === +qidx));
  const r = DATA.find(x => x.qidx === +qidx);
  if (!r) return;
  const h = $('qheader');
  h.classList.remove('empty');
  h.innerHTML = `
    <img src="${r.qpath}">
    <div class="info">
      <h2 style="margin:0 0 8px">Query #${r.qidx}</h2>
      <div><b>true id:</b> ${r.qpid}</div>
      <div><b>camera:</b> c${r.qcam}</div>
      <div><b>class:</b> ${r.qcls}</div>
      <div><b>matches in top-${r.top.length}:</b> ${r.n_match_in_topK} / ${r.n_total_match}</div>
      <div><b>rerank params:</b> k1=${r.params_used.k1}, k2=${r.params_used.k2}, λ=${r.params_used.lam}</div>
      <div style="color:#888;margin-top:6px">${r.qpath}</div>
    </div>`;
  const g = $('grid');
  g.innerHTML = '';
  for (const t of r.top) {
    const card = document.createElement('div');
    card.className = 'card ' + (t.match ? 'match' : 'nomatch');
    card.innerHTML = `
      <img src="${t.path}">
      <div class="caption">
        <span class="rank">#${t.rank}</span>
        <span class="badge ${t.match ? 'match' : 'nomatch'}">${t.match ? 'MATCH' : 'no'}</span>
        <br>
        <b>id ${t.pid}</b> · c${t.cam} · ${t.cls}<br>
        d=${t.rerank_dist.toFixed(3)} sim=${t.cos_sim.toFixed(3)}
      </div>`;
    g.appendChild(card);
  }
}

renderSidebar();
$('search').oninput = (e) => renderSidebar(e.target.value);
</script>
</body></html>
"""
    html = html.replace('TAG_PLACEHOLDER', args.tag)
    html = html.replace('DATA_PLACEHOLDER', json.dumps(records))

    out_path = os.path.join(REPO, args.out)
    with open(out_path, 'w') as f:
        f.write(html)
    print(f'wrote {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB)')
    print(f'open in browser:  file://{out_path}')


if __name__ == '__main__':
    main()
