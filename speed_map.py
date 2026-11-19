#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speed_map.py — карта швидкості по ділянках маршруту: зелений = їдуть швидко, червоний = стоять/повзуть.

Лежить поруч з analyze1.py (використовує його функції) і читає ті самі positions-РРРР-ММ-ДД.csv.gz.
Результат — ОДИН html-файл, який відкривається у браузері (потрібен інтернет лише для підкладки карти).

Приклади:
    python speed_map.py --data D:\\transit-data --list-routes
    # УСІ маршрути (за замовчуванням), справжні траси з папки lines, назви кінцевих зі static:
    python speed_map.py --data D:\\transit-data --static D:\\static --lines D:\\lines --out speedmap
    # лише деякі:
    python speed_map.py --data D:\\transit-data --static D:\\static --lines D:\\lines --routes Т03,Т08,А03а
    python speed_map.py --data D:\\transit-data --static D:\\static --lines D:\\lines --window ранок --show-all

Як рахується:
  • траса маршруту ріжеться на ділянки по --bin-m метрів (за замовчуванням 100);
  • кожна точка ТЗ проєктується на трасу свого напрямку, береться середня швидкість у точках ділянки;
  • кінцеві зони (перші/останні 150 м) не беруться — там стоянка на кінцевій, не затор;
  • ділянки, де менше --min-n точок, сірі («мало даних»);
  • під кольоровими ділянками малюється сама траса (біла обводка) — видно весь маршрут, навіть де даних мало.

У карті: праворуч вгорі перемикач підкладки (OSM, світла, темна, супутник, топо...) і список маршрутів
(згруповані: автобуси / тролейбуси / трамваї; галочка типу вмикає всю групу, є «усі / жодного» і пошук).
Без --show-all при «усіх» маршрутах спочатку увімкнено лише перший маршрут кожного типу.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import analyze1 as A

OFFSET_M = 6            # зсув ліній двох напрямків убік (праворуч за ходом), щоб не накладалися
PROBE_KEYS = 200        # скільки «ТЗ-днів» брати для визначення, яка траса відповідає напряму
COLORS = [(0.0, (215, 48, 39)), (0.5, (254, 224, 80)), (1.0, (26, 152, 80))]   # червоний → жовтий → зелений
GRAY = '#9e9e9e'


def color_of(v, vmin, vmax):
    t = min(1.0, max(0.0, (v - vmin) / (vmax - vmin)))
    for (t0, c0), (t1, c1) in zip(COLORS[:-1], COLORS[1:]):
        if t <= t1:
            k = (t - t0) / (t1 - t0)
            return '#%02x%02x%02x' % tuple(int(round(a + (b - a) * k)) for a, b in zip(c0, c1))
    return '#%02x%02x%02x' % COLORS[-1][1]


def forward_share(S, g):
    """Частка рухомих пар точок, де ТЗ просувається ВПЕРЕД по трасі S (g відсортовано за raw_id, ts)."""
    s, d = S.project(g.lat.values, g.lon.values)
    ok = d <= A.D_MAX_M
    same = g.raw_id.values[1:] == g.raw_id.values[:-1]
    dt, ds = np.diff(g.ts.values), np.diff(s)
    m = same & ok[1:] & ok[:-1] & (dt > 0) & (dt <= 120) & (g.speed.values[1:] > 5) & (np.abs(ds) < 500)
    if m.sum() < 20:
        return 0.0, float(ok.mean())
    return float((ds[m] > 0).mean()), float(ok.mean())


def pick_shape(shapes, g):
    """Серед кандидатів обираємо трасу, вздовж якої ТЗ цього напрямку справді їдуть вперед і лежать на ній."""
    keys = (g.raw_id + '|' + g.day).drop_duplicates().head(PROBE_KEYS)
    probe = g[(g.raw_id + '|' + g.day).isin(set(keys))]
    best, best_score = None, 0.0
    for sid, sdir, S in shapes:
        f, cov = forward_share(S, probe)
        if f * cov > best_score:
            best, best_score = (sid, S), f * cov
    return best, best_score


def trip_keys(g):
    """Номер «поїздки» для кожного рядка (g відсортовано за raw_id, ts). Розрив: інший ТЗ/день, пропуск даних > 10 хв
    або довга стоянка на місці (≥ 4 хв — оборот на кінцевій): тоді «туди» й «назад» не злипаються в одну поїздку."""
    r, ts, day = g.raw_id.values, g.ts.values, g.day.values
    spd = g.speed.fillna(0).values
    brk = np.r_[True, (r[1:] != r[:-1]) | (day[1:] != day[:-1]) | (np.diff(ts) > 600)]
    slow, n, i = spd < 3, len(ts), 0
    while i < n:
        if slow[i]:
            j = i
            while j + 1 < n and slow[j + 1] and not brk[j + 1]:
                j += 1
            if ts[j] - ts[i] >= 240 and j + 1 < n:
                brk[j + 1] = True
            i = j + 1
        else:
            i += 1
    return np.cumsum(brk) - 1


def _variant_score(S, lat, lon, spd, ts):
    return A.variant_score(S, lat, lon, spd, ts)


def pick_variants(shapes, g, min_score=0.5, min_on=0.5, min_fwd=0.6):
    """Для кожної поїздки обирає варіант траси, чинний у цей день.
    1) повний збіг: частка на трасі × охоплення × рух уздовж ≥ min_score (продовжений маршрут: перемагає варіант, що відповідає всьому шляху);
    2) інакше неповний рейс (скорочений, частковий): ≥ min_on точок на лінії і ≥ min_fwd руху вздовж її напрямку.
    Повертає (мітки по рядках: індекс у shapes або -1, список Shape, невідповідні поїздки).
    Невідповідна поїздка: (a, b, причина, частка на лінії, охоплення, рух уздовж, точок на лінії) за найкращою лінією;
    причина: 1 — надто мало точок на лінії (<8), 2 — поза лінією (гараж, об'їзд, інший шлях),
    3 — немає чинного варіанта в цей день, 4 — по лінії, але рух не вздовж її напрямку (мішанина напрямів, петля)."""
    cands = [S for _, _, S in shapes]
    tk = trip_keys(g)
    lab = np.full(len(g), -1)
    unm = []
    lat, lon, spd, ts, day = g.lat.values, g.lon.values, g.speed.fillna(0).values, g.ts.values, g.day.values
    starts = np.flatnonzero(np.r_[True, tk[1:] != tk[:-1]])
    for a_, b_ in zip(starts, np.r_[starts[1:], len(g)]):
        if b_ - a_ < 8:
            unm.append((a_, b_, 1, 0.0, 0.0, 0.0, 0))
            continue
        valid = [(i, S) for i, S in enumerate(cands) if A.shape_valid(S, day[a_])]
        if not valid:
            unm.append((a_, b_, 3, 0.0, 0.0, 0.0, 0))
            continue
        best, bs = -1, min_score
        alt, alt_sc, alt_cov = -1, 0.0, -1.0
        bp = (-1.0, 0.0, 0.0, 0)
        for i, S in valid:
            s_, d_ = S.project(lat[a_:b_], lon[a_:b_])
            on_f, cover, fwd, n_on = A.variant_parts(S, s_, d_, spd[a_:b_], ts[a_:b_])
            if on_f > bp[0]:
                bp = (on_f, cover, fwd, n_on)
            if n_on < 8:
                continue
            sc = on_f * cover * fwd
            if sc > bs + 1e-9:
                best, bs = i, sc
            if on_f >= min_on and fwd >= min_fwd:
                a2 = on_f * fwd
                if a2 > alt_sc + 1e-9 or (abs(a2 - alt_sc) <= 1e-9 and cover > alt_cov):
                    alt, alt_sc, alt_cov = i, a2, cover
        if best < 0 and alt >= 0:
            best = alt
        if best < 0:
            code = 1 if (bp[0] >= min_on and bp[3] < 8) else (2 if bp[0] < min_on else 4)
            unm.append((a_, b_, code) + bp)
        lab[a_:b_] = best
    return lab, cands, unm


def _normals(S):
    L = np.maximum(S.seglen, 1e-9)
    return np.c_[S.dy / L, -S.dx / L]           # нормаль праворуч за ходом кожного сегмента


def seg_coords(S, nrm, s0, s1, off):
    """Вершини ламаної між s0 і s1 (м вздовж траси), зсунуті на off м праворуч; повертає [[lat, lon], ...]."""
    nseg = len(S.seglen)
    pts = [s0] + [c for c in S.cum if s0 < c < s1] + [s1]
    out = []
    for s in pts:
        j = int(np.clip(np.searchsorted(S.cum, s, 'right') - 1, 0, nseg - 1))
        t = (s - S.cum[j]) / max(S.seglen[j], 1e-9)
        x, y = S.x[j] + t * S.dx[j], S.y[j] + t * S.dy[j]
        n = nrm[j]
        if abs(s - S.cum[j]) < 1e-6 and j > 0:          # вершина між двома сегментами — усереднюємо нормалі
            n = nrm[j] + nrm[j - 1]
            n = n / max(np.hypot(*n), 1e-9)
        out.append([round(S.lat0 + (y + n[1] * off) / S.ky, 6), round(S.lon0 + (x + n[0] * off) / S.kx, 6)])
    return out


def build_layer(route, dr, S, g, a):
    s, d = S.project(g.lat.values, g.lon.values)
    keep = (d <= a.dmax) & (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M) & g.speed.notna().values
    if keep.sum() == 0:
        return None, [], None, None
    nb = int(math.ceil(S.L / a.bin_m))
    b = np.minimum((s[keep] // a.bin_m).astype(int), nb - 1)
    v = g.speed.values[keep].astype(float)
    cnt = np.bincount(b, minlength=nb)
    mean = np.bincount(b, weights=v, minlength=nb) / np.maximum(cnt, 1)
    slow = np.bincount(b, weights=(v < 3).astype(float), minlength=nb) / np.maximum(cnt, 1)
    nrm = _normals(S)
    segs, rows = [], []
    for i in range(nb):
        s0, s1 = i * a.bin_m, min((i + 1) * a.bin_m, S.L)
        if s1 - s0 < 5:
            continue
        coords = seg_coords(S, nrm, s0, s1, OFFSET_M)
        if cnt[i] >= a.min_n:
            col = color_of(mean[i], a.vmin, a.vmax)
            tip = f"{route} {A.dir_label(route, dr)}<br>ділянка {int(s0)}–{int(s1)} м<br>" \
                  f"<b>{mean[i]:.1f} км/год</b> (стоїть: {100 * slow[i]:.0f}% часу)<br>точок: {int(cnt[i])}"
        else:
            col = GRAY
            tip = f"{route} {A.dir_label(route, dr)}<br>ділянка {int(s0)}–{int(s1)} м<br>мало даних (точок: {int(cnt[i])})"
        segs.append([col, coords, tip])
        rows.append({'route': route, 'dir': dr, 'direction': A.dir_label(route, dr), 'from_m': int(s0), 'to_m': int(s1),
                     'points': int(cnt[i]), 'mean_speed_kmh': round(float(mean[i]), 1) if cnt[i] else None,
                     'stopped_share': round(float(slow[i]), 2) if cnt[i] else None,
                     'lat': coords[0][0], 'lon': coords[0][1]})
    stats = {'mean': round(float(v.mean()), 1), 'moving': round(float(v[v >= 3].mean()), 1) if (v >= 3).any() else None,
             'stopped': round(100 * float((v < 3).mean())), 'n': int(len(v)), 'km': round(S.L / 1000, 1)}
    line = seg_coords(S, nrm, 0.0, float(S.L), OFFSET_M)      # уся траса (для обводки під кольоровими ділянками)
    return segs, rows, line, stats


KIND_TITLES = {'bus': '🚌 Автобуси', 'trolley': '🚎 Тролейбуси', 'tram': '🚋 Трамваї'}
KIND_ALIASES = {'bus': 'bus', 'автобус': 'bus', 'автобуси': 'bus', 'a': 'bus',
                'trolley': 'trolley', 'trolleybus': 'trolley', 'тролейбус': 'trolley', 'тролейбуси': 'trolley', 't': 'trolley',
                'tram': 'tram', 'трамвай': 'tram', 'трамваї': 'tram'}


def kind_of(vt, route):
    """Тип ТЗ: bus / trolley / tram — за полем vtype (текст або GTFS route_type), інакше за назвою маршруту."""
    t = str(vt).strip().lower()
    if 'трам' in t or 'tram' in t or t in ('0', '900'):
        return 'tram'
    if 'тро' in t or 'trol' in t or t in ('11', '800'):
        return 'trolley'
    if 'авт' in t or 'bus' in t or t in ('3', '700'):
        return 'bus'
    r = str(route).strip()
    if r.isdigit():
        return 'tram'
    if r[:1] in ('Т', 'T', 'т', 't'):
        return 'trolley'
    return 'bus'


HTML = """<!doctype html>
<html lang="uk"><head><meta charset="utf-8"><title>Швидкість по ділянках</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
html,body,#map{height:100%;margin:0} body{font-family:system-ui,Arial,sans-serif}
.legend{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:12px;line-height:1.4}
.bar{height:10px;width:180px;border-radius:3px;background:linear-gradient(to right,__GRAD__);margin:4px 0}
.row{display:flex;justify-content:space-between}
.sw{display:inline-block;width:12px;height:10px;background:#9e9e9e;margin-right:4px;vertical-align:middle}
.leaflet-control-layers-list{max-height:calc(100vh - 190px);overflow-y:auto;overflow-x:hidden}
.tools{background:#fff;padding:6px 8px;border-radius:6px;box-shadow:0 1px 6px rgba(0,0,0,.3);font-size:12px}
.tools button{margin-right:4px;cursor:pointer}
.panel{max-height:calc(100vh - 330px);overflow-y:auto;overflow-x:hidden;margin-top:6px;width:230px}
.kind{margin-bottom:4px} .kind>.kh{display:flex;align-items:center;gap:4px;font-weight:600}
.kind .fold{cursor:pointer;margin-left:auto;user-select:none}
.rl{display:flex;flex-wrap:wrap;gap:2px 10px;margin:2px 0 4px 18px}
.rl label{white-space:nowrap;cursor:pointer}
.tools input{width:150px;margin-top:5px;box-sizing:border-box}
</style></head><body><div id="map"></div>
<script>
const DATA = __DATA__;
const map = L.map('map', {preferCanvas: true});

const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 19, attribution: '&copy; OpenStreetMap'});
const light = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, subdomains: 'abcd', attribution: '&copy; OpenStreetMap &copy; CARTO'});
const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {maxZoom: 19, subdomains: 'abcd', attribution: '&copy; OpenStreetMap &copy; CARTO'});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Tiles &copy; Esri'});
const satLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Labels &copy; Esri'});
const esriTopo = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Tiles &copy; Esri'});
const otopo = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
  {maxZoom: 17, subdomains: 'abc', attribution: '&copy; OpenStreetMap, SRTM | &copy; OpenTopoMap'});

const bases = {
  'Карта OSM': osm,
  'Світла (CARTO)': light,
  'Темна (CARTO)': dark,
  'Супутник + підписи': L.layerGroup([sat, satLabels]),
  'Супутник': sat,
  'Топографічна (Esri)': esriTopo,
  'Топографічна (OpenTopoMap, рельєф)': otopo
};
osm.addTo(map);

const KINDS = {bus: '🚌 Автобуси', trolley: '🚎 Тролейбуси', tram: '🚋 Трамваї'};
const routes = {};          // маршрут -> {g, kind, pts, cb}
DATA.layers.forEach(l => {
  let R = routes[l.route];
  if (!R) R = routes[l.route] = {g: L.layerGroup(), kind: l.kind, pts: [], visible: false, dirs: []};
  R.dirs.push({name: l.name, st: l.stats});
  if (l.line && l.line.length > 1) {        // сама траса: біла обводка під кольоровими ділянками
    L.polyline(l.line, {color: '#ffffff', weight: 10, opacity: 0.85, lineCap: 'round', interactive: false}).addTo(R.g);
    l.line.forEach(p => R.pts.push(p));
  }
  l.segs.forEach(sg => {
    L.polyline(sg[1], {color: sg[0], weight: 6, opacity: 0.95, lineCap: 'butt'})
      .bindTooltip(sg[2], {sticky: true}).addTo(R.g);
  });
  if (l.visible) R.visible = true;
});
L.control.layers(bases, null, {collapsed: false, position: 'topright'}).addTo(map);

const bounds = [];
Object.values(routes).forEach(R => { if (R.visible) { R.g.addTo(map); R.pts.forEach(p => bounds.push(p)); } });
if (bounds.length) map.fitBounds(bounds); else map.setView([49.84, 24.03], 12);

const tools = L.control({position: 'topright'});
tools.onAdd = () => {
  const d = L.DomUtil.create('div', 'tools');
  d.innerHTML = '<button id="all">усі</button><button id="none">жодного</button><br>'
    + '<input id="q" placeholder="пошук маршруту..."><div class="panel" id="panel"></div>';
  L.DomEvent.disableClickPropagation(d);
  L.DomEvent.disableScrollPropagation(d);
  return d;
};
tools.addTo(map);

const kindCb = {};
function renderStats() {
  const box = document.getElementById('stats');
  const rows = [];
  Object.keys(routes).forEach(n => {
    if (!routes[n].cb || !routes[n].cb.checked) return;
    routes[n].dirs.forEach(d => {
      const st = d.st;
      rows.push('<div style="margin:3px 0"><b>' + d.name + '</b><br>середня <b>' + st.mean + ' км/год</b>'
        + (st.moving !== null ? ' (в русі ' + st.moving + ')' : '') + ', стоїть ' + st.stopped + '% часу<br>'
        + '<span style="color:#666">траса ' + st.km + ' км, точок ' + st.n + '</span></div>');
    });
  });
  box.style.display = rows.length ? '' : 'none';
  box.innerHTML = '<b>Середня швидкість</b>' + rows.join('');
}
function setRoute(name, on) {
  const R = routes[name];
  if (on) R.g.addTo(map); else map.removeLayer(R.g);
  R.cb.checked = on;
  renderStats();
}
function syncKind(k) {
  const rs = Object.values(routes).filter(R => R.kind === k);
  const n = rs.filter(R => R.cb.checked).length;
  kindCb[k].checked = n === rs.length;
  kindCb[k].indeterminate = n > 0 && n < rs.length;
}
const panel = document.getElementById('panel');
Object.keys(KINDS).forEach(k => {
  const names = Object.keys(routes).filter(n => routes[n].kind === k);
  if (!names.length) return;
  const box = document.createElement('div'); box.className = 'kind';
  const head = document.createElement('div'); head.className = 'kh';
  const kc = document.createElement('input'); kc.type = 'checkbox'; kindCb[k] = kc;
  const lab = document.createElement('span'); lab.textContent = KINDS[k] + ' (' + names.length + ')';
  const fold = document.createElement('span'); fold.className = 'fold'; fold.textContent = '▾';
  head.append(kc, lab, fold);
  const list = document.createElement('div'); list.className = 'rl';
  names.forEach(n => {
    const lb = document.createElement('label');
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = routes[n].visible;
    routes[n].cb = cb;
    lb.title = routes[n].dirs.map(d => d.name + ': ' + d.st.mean + ' км/год').join(String.fromCharCode(10));
    cb.onchange = () => { setRoute(n, cb.checked); syncKind(k); };
    lb.append(cb, ' ' + n);
    list.append(lb);
  });
  kc.onchange = () => { names.forEach(n => setRoute(n, kc.checked)); syncKind(k); };
  fold.onclick = () => { const h = list.style.display === 'none'; list.style.display = h ? '' : 'none'; fold.textContent = h ? '▾' : '▸'; };
  box.append(head, list); panel.append(box);
  syncKind(k);
});
document.getElementById('all').onclick = () => { Object.keys(routes).forEach(n => setRoute(n, true)); Object.keys(kindCb).forEach(syncKind); };
document.getElementById('none').onclick = () => { Object.keys(routes).forEach(n => setRoute(n, false)); Object.keys(kindCb).forEach(syncKind); };
document.getElementById('q').oninput = e => {
  const q = e.target.value.trim().toLowerCase();
  panel.querySelectorAll('.rl label').forEach(lb => {
    lb.style.display = (!q || lb.textContent.toLowerCase().includes(q)) ? '' : 'none';
  });
};

const sc = L.control({position: 'bottomright'});
sc.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'stats';
  d.style.cssText = 'max-height:40vh;overflow-y:auto;max-width:300px'; L.DomEvent.disableScrollPropagation(d); return d; };
sc.addTo(map);
renderStats();

const lg = L.control({position: 'bottomleft'});
lg.onAdd = () => { const d = L.DomUtil.create('div', 'legend');
  d.innerHTML = '<b>__TITLE__</b><div class="bar"></div><div class="row"><span>__VMIN__</span><span>__VMID__</span><span>__VMAX__ км/год</span></div>'
    + '<div><span class="sw"></span>мало даних</div>'; return d; };
lg.addTo(map);
</script></body></html>
"""


def write_html(layers, title, a, path):
    mid = (a.vmin + a.vmax) / 2
    grad = ','.join(color_of(a.vmin + (a.vmax - a.vmin) * k / 10, a.vmin, a.vmax) for k in range(11))
    html = (HTML.replace('__DATA__', json.dumps({'layers': layers}, ensure_ascii=False, separators=(',', ':')))
            .replace('__GRAD__', grad).replace('__TITLE__', title)
            .replace('__VMIN__', f"{a.vmin:g}").replace('__VMID__', f"{mid:g}").replace('__VMAX__', f"{a.vmax:g}"))
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def main():
    ap = argparse.ArgumentParser(description='Карта швидкості по ділянках маршруту')
    ap.add_argument('--data', default='.', help='папка з positions-*.csv.gz')
    ap.add_argument('--out', default='speedmap')
    ap.add_argument('--routes', default='all',
                    help='маршрути через кому, як у даних (Т03,Т08,А03а), або all / порожньо = усі (за замовчуванням)')
    ap.add_argument('--static', default='', help='static.zip (або папка з static-*.zip): траси маршрутів і назви кінцевих')
    ap.add_argument('--routes-dir', '--lines', dest='routes_dir', default='',
                    help='папка lines з трасами <маршрут>_line.json (мають пріоритет над static)')
    ap.add_argument('--start-zone', type=float, default=150, help=argparse.SUPPRESS)   # приймається для сумісності, не використовується
    ap.add_argument('--end-zone', type=float, default=120, help=argparse.SUPPRESS)
    ap.add_argument('--plan', default='', help=argparse.SUPPRESS)
    ap.add_argument('--detour', default='', help="період об'їзду РРРР-ММ-ДД:РРРР-ММ-ДД")
    ap.add_argument('--detour-routes', default='', help="маршрути об'їзду через кому (порожньо = усі)")
    ap.add_argument('--days', default='', help='обмежити днями через кому: 2026-11-10,2026-11-11')
    ap.add_argument('--day-types', default='', help='CSV day,type для свят і особливих днів')
    ap.add_argument('--day-type', default='будній', help='будній / субота / неділя / свято / weekday_detour (дні об’їзду) / all')
    ap.add_argument('--window', default='всі', help='ранок / обід / вечір / всі (6:00–22:00)')
    ap.add_argument('--bin-m', type=int, default=100, help='довжина ділянки, м')
    ap.add_argument('--min-n', type=int, default=15, help='мінімум точок на ділянці, інакше сірий колір')
    ap.add_argument('--vmin', type=float, default=5, help='швидкість (км/год), що вважається «червоною»')
    ap.add_argument('--vmax', type=float, default=30, help='швидкість (км/год), що вважається «зеленою»')
    ap.add_argument('--dmax', type=float, default=40, help='макс. відстань точки від траси, м')
    ap.add_argument('--show-all', action='store_true',
                    help='одразу показати всі маршрути (інакше при «усіх» увімкнено лише перший)')
    ap.add_argument('--types', default='all', help='типи ТЗ через кому: bus,trolley,tram (або автобус,тролейбус,трамвай); all = усі')
    ap.add_argument('--split-types', action='store_true', help='окрім спільної карти, зробити окремі html: speed_map_bus.html тощо')
    ap.add_argument('--list-routes', action='store_true')
    a = ap.parse_args()

    only_days = {d.strip() for d in a.days.split(',') if d.strip()}
    print("Завантаження:")
    df = A.load_positions(a.data, only_days)
    overrides = A.load_day_overrides(a.day_types)
    df['day_type'] = A.assign_day_types(df, overrides, a.detour, a.detour_routes)

    if a.list_routes:
        g = df.groupby('route').agg(vehicles=('raw_id', 'nunique'), vtype=('vtype', lambda s: s.mode().iloc[0]))
        g['kind'] = [kind_of(v, r) for r, v in zip(g.index, g.vtype)]
        print(g.sort_index().to_string())
        return

    dt_name = A.DAY_ALIASES.get(a.day_type.strip().lower())
    if a.day_type.strip().lower() not in ('all', 'всі') and dt_name:
        df = df[df.day_type == dt_name]
    win = a.window.strip().lower()
    if win in A.WINDOWS:
        lo, hi = A.WINDOWS[win]
        df = df[(df.hour >= lo) & (df.hour < hi)]
    else:
        df = df[(df.hour >= 6) & (df.hour < 22)]
    if df.empty:
        sys.exit("Після фільтрів (тип дня / частина доби) даних не лишилось.")
    print(f"Дні в аналізі: {', '.join(sorted(df.day.unique()))}")

    if a.static:
        A.load_dir_names(a.static, df)
    own = A.load_routes_dir(a.routes_dir) if a.routes_dir else {}
    static = A.load_static(a.static) if a.static else {}
    if not own and not static:
        sys.exit("Потрібні траси маршрутів: --static і/або --lines (--routes-dir)")

    all_mode = a.routes.strip().lower() in ('', 'all', 'всі', '*')
    if all_mode:
        routes = sorted(df.route.unique())
    else:
        routes = A.resolve_routes([x for x in a.routes.split(',')], sorted(df.route.unique()))
    vt = df.groupby('route').vtype.agg(lambda s_: s_.mode().iloc[0])
    kinds = {r: kind_of(vt.get(r, ''), r) for r in routes}
    want = {x.strip().lower() for x in a.types.split(',') if x.strip()}
    if want and not ({'all', 'всі'} & want):
        wk = {KIND_ALIASES[w] for w in want if w in KIND_ALIASES}
        routes = [r for r in routes if kinds[r] in wk]
    print(f"Значення vtype у даних: {sorted(map(str, df.vtype.unique()))}")
    for k, t in KIND_TITLES.items():
        print(f"  {t}: {sum(1 for r in routes if kinds[r] == k)}")
    print(f"Маршрути ({len(routes)}): {', '.join(routes)}")

    layers, all_rows, skipped = [], [], []
    for r in routes:
        shapes = (A.get_shapes(own, r) if own else None) or (A.get_shapes(static, r) if static else None)
        if not shapes:
            print(f"  {r}: траси немає у --static/--lines — пропускаю")
            skipped.append(r)
            continue
        sub = df[df.route == r].dropna(subset=['dir'])
        for dr, g in sub.groupby('dir'):
            g = g.sort_values(['raw_id', 'ts']).reset_index(drop=True)
            best, score = pick_shape(shapes, g)
            if best is None or score < 0.2:
                print(f"  {r}, напрям {int(dr)}: не вдалося підібрати трасу (точок {len(g)}) — пропускаю")
                continue
            segs, rows, line, stats = build_layer(r, dr, best[1], g, a)
            if not segs:
                print(f"  {r}, напрям {int(dr)}: немає точок біля траси")
                continue
            ok = sum(1 for s_ in segs if s_[0] != GRAY)
            print(f"  {r} {A.dir_label(r, dr)}: ділянок {len(segs)}, з даними {ok}, точок {len(g)}, середня {stats['mean']} км/год")
            layers.append({'name': f"{r} {A.dir_label(r, dr)}", 'route': r, 'kind': kinds[r], 'line': line, 'stats': stats, 'segs': segs})
            all_rows += rows
    if not layers:
        sys.exit("Нічого малювати.")

    # які шари увімкнені на старті: перший маршрут кожного типу (або все, якщо --show-all / вказано --routes)
    first_of = {}
    for l in layers:
        first_of.setdefault(l['kind'], l['route'])
    for l in layers:
        l['visible'] = bool(a.show_all or not all_mode or first_of[l['kind']] == l['route'])
    os.makedirs(a.out, exist_ok=True)
    title = f"Швидкість, {a.day_type}, {a.window}"
    p = write_html(layers, title, a, os.path.join(a.out, 'speed_map.html'))
    if a.split_types:
        for k in KIND_TITLES:
            sub = [l for l in layers if l['kind'] == k]
            if sub:
                write_html(sub, f"{title} · {KIND_TITLES[k]}", a, os.path.join(a.out, f'speed_map_{k}.html'))
    pd.DataFrame(all_rows).to_csv(os.path.join(a.out, 'speed_segments.csv'), index=False)
    pd.DataFrame([{'route': l['route'], 'kind': l['kind'], 'direction': l['name'], **l['stats']} for l in layers]).rename(
        columns={'mean': 'mean_speed_kmh', 'moving': 'moving_speed_kmh', 'stopped': 'stopped_pct', 'n': 'points', 'km': 'length_km'}
    ).to_csv(os.path.join(a.out, 'speed_routes.csv'), index=False)
    print(f"\nМаршрутів на карті: {len({l['route'] for l in layers})}, шарів (напрямків): {len(layers)}"
          + (f", без траси: {', '.join(skipped)}" if skipped else ""))
    print(f"Готово: {os.path.abspath(p)}\nТаблиця ділянок: {os.path.abspath(os.path.join(a.out, 'speed_segments.csv'))}")


if __name__ == '__main__':
    main()
