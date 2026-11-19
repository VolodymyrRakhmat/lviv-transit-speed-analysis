#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py — аналітика за експортованими даними колектора (positions-РРРР-ММ-ДД.csv.gz).

Запускається на вашому ПК (або на сервері) у папці з експортами. Потрібно: pandas, numpy, matplotlib
    pip install pandas numpy matplotlib

Приклади:
    python analyze.py --data D:\\transit-data --out report
    python analyze.py --data D:\\transit-data --out report --route all
    python analyze.py --data D:\\transit-data --out report --route А03 --plan plan.csv
    python analyze.py --data D:\\transit-data --list-routes

Що рахується (папка --out):
    days.csv               які дні взяті в аналіз і чи повні вони
    vehicles_by_window.csv скільки ТЗ у середньому їде на маршруті зранку / в обід / ввечері
    peak_by_route.csv      піковий одночасний випуск по маршрутах
    trip_times.csv         час рейсу (повний і «у русі») за напрямком та частиною дня
    cycle_interval.csv     час кола і оцінка інтервалу руху (коло / кількість ТЗ)
    speeds_by_route.csv, speeds_by_hour.csv
    plan_vs_fact.csv       (якщо задано --plan) виконання міського плану випуску
    report.md              короткий текстовий підсумок
    *.png                  графіки

ВАЖЛИВО ПРО МЕТОД ЧАСУ РЕЙСУ. Рейс визначається за траєкторією ТЗ вздовж лінії маршруту (потрібні --static
і/або --routes-dir): від рушання з початкової до приїзду на кінцеву, стоянка на кінцевій не входить. Беруться лише
повні рейси (≥ 85% траси), неповні відрізки та уривки в статистику не потрапляють. trip_id для часу рейсу НЕ
використовується: у фіді він перемикається не за фактом проходження рейсу (дуже багато серій коротше 3 хв).
"""
import argparse
import glob
import os
import json
import re
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

WINDOWS = {'ранок': (6, 9), 'обід': (11, 14), 'вечір': (16, 19)}   # [від, до) години за Києвом
GAP_SEC = 300            # розрив у спостереженні, після якого відрізок вважається перерваним
MIN_TRIP_MIN, MAX_TRIP_MIN = 4, 180
ZONE_M = 150             # «стартова/кінцева зона» траси (м): вимірюємо час між виходом із першої і входом в останню
D_MAX_M = 80             # точка вважається «на трасі», якщо ближче цієї відстані до лінії маршруту
BACK_TOL_M = 500         # якщо ТЗ відкотилось назад по трасі більше — це не наскрізний проїзд
                         # (500 м, бо на «петлях» і відгалуженнях траси проєкція може стрибати)
COMPLETE_SHARE = 0.95    # частка 5-хв інтервалів 06:00-22:00 з даними, щоб день вважався повним
DAY_TYPE_UA = {'weekday': 'будній', 'saturday': 'субота', 'sunday': 'неділя', 'holiday': 'свято'}
for _k, _v in list(DAY_TYPE_UA.items()):          # окремі типи днів для періоду об'їзду: weekday_detour тощо
    DAY_TYPE_UA[_k + '_detour'] = _v + ' (об’їзд)'
REPORT_CONSOLE_LINES = 60   # скільки рядків звіту друкувати в консоль (повний — у report.md)

# ----------------------------- назви маршрутів ---------------------------
_LAT2CYR = str.maketrans({'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М', 'O': 'О',
                          'P': 'Р', 'T': 'Т', 'X': 'Х', 'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р',
                          'x': 'х', 'r': 'р', 't': 'т'})
VTYPE_ALIASES = {'bus': 'bus', 'автобус': 'bus', 'tram': 'tram', 'трамвай': 'tram',
                 'trolley': 'trolley', 'trolleybus': 'trolley', 'тролейбус': 'trolley'}
DAY_ALIASES = {'weekday': 'weekday', 'будній': 'weekday', 'будні': 'weekday', 'saturday': 'saturday',
               'субота': 'saturday', 'sunday': 'sunday', 'неділя': 'sunday', 'holiday': 'holiday',
               'свято': 'holiday', 'святковий': 'holiday'}
for _k in ('weekday', 'saturday', 'sunday', 'holiday'):
    DAY_ALIASES[_k + '_detour'] = _k + '_detour'


def parse_route(text, vtype=None):
    t = (str(text) if text is not None else '').strip().translate(_LAT2CYR)
    low = t.lower()
    if low.startswith('тр') or low.startswith('tr'):
        vt = 'trolley'
    elif low.startswith('т'):
        vt = 'tram'
    elif low.startswith('а'):
        vt = 'bus'
    else:
        vt = None
    vt = vtype or vt
    m = re.search(r'(\d+)', t)
    if not m:
        return (vt, None, low)
    return (vt, int(m.group(1)), t[m.end():].strip().lower())


def route_match(a, b):
    return (a[1] is not None and a[1] == b[1] and a[2] == b[2]
            and (a[0] is None or b[0] is None or a[0] == b[0]))


def resolve_routes(tokens, available):
    """Користувач може написати «А03», «3» або «3а» — підбираємо назви з даних."""
    found = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if tok in available:
            found.append(tok)
            continue
        p = parse_route(tok)
        cand = [r for r in available if route_match(p, parse_route(r))]
        if not cand and p[1] is not None:
            cand = [r for r in available if parse_route(r)[1] == p[1]]
            if cand:
                print(f"«{tok}»: точного збігу немає, схожі в даних: {', '.join(sorted(cand))}")
        if not cand:
            print(f"«{tok}»: у даних такого маршруту не знайдено (див. --list-routes)")
        found.extend(cand)
    return sorted(set(found))


# ----------------------------- завантаження ------------------------------
def load_positions(folder, only_days):
    frames = []
    for f in sorted(glob.glob(os.path.join(folder, 'positions-*'))):
        if not (f.endswith('.csv') or f.endswith('.csv.gz')):
            continue
        m = re.search(r'positions-(\d{4}-\d{2}-\d{2})', os.path.basename(f))
        if not m or (only_days and m.group(1) not in only_days):
            continue
        frames.append(pd.read_csv(f, dtype={'raw_id': str, 'plate': str, 'route': str}))
        print(f"  завантажено {os.path.basename(f)}: {len(frames[-1])} рядків")
    if not frames:
        sys.exit(f"У папці {folder} немає файлів positions-РРРР-ММ-ДД.csv(.gz)")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=['trip_id'], errors='ignore')        # trip_id фіду в аналізі не використовується
    df = df.drop_duplicates(subset=['raw_id', 'ts'])
    df['slot'] = df['hour'] * 12 + (df['ts'] % 3600) // 300
    return df


def load_day_overrides(path):
    out = {}
    if path:
        for _, r in pd.read_csv(path, dtype=str).iterrows():
            t = DAY_ALIASES.get(str(r['type']).strip().lower())
            if t:
                out[str(r['day']).strip()] = t
    return out


def day_type_of(day, overrides):
    if day in overrides:
        return overrides[day]
    wd = datetime.strptime(day, '%Y-%m-%d').weekday()
    return 'weekday' if wd < 5 else ('saturday' if wd == 5 else 'sunday')


def assign_day_types(df, overrides, detour='', detour_routes='', default_vt='bus'):
    """Тип дня для кожного рядка. Для змінених маршрутів у період об'їзду тип стає '<тип>_detour'
    (weekday_detour тощо) — такі дані рахуються окремо і не змішуються зі звичайними днями.
    detour = '2026-11-13:2026-11-22' (кілька періодів — через кому); detour_routes = '17,46,55а,84,...'
    (порожньо = усі маршрути). Номери без літери-префікса вважаються автобусними (default_vt)."""
    base = df['day'].map(lambda d: day_type_of(d, overrides))
    spans = []
    for part in str(detour or '').split(','):
        part = part.strip()
        if part:
            d0, _, d1 = part.partition(':')
            spans.append((d0.strip(), (d1 or d0).strip()))
    if not spans:
        return base
    mask = np.zeros(len(df), bool)
    for d0, d1 in spans:
        mask |= ((df['day'] >= d0) & (df['day'] <= d1)).values
    if str(detour_routes or '').strip():
        avail, hit = sorted(df.route.unique()), set()
        for tok in detour_routes.split(','):
            tok = tok.strip()
            if not tok:
                continue
            p = parse_route(tok)
            if p[0] is None:
                p = (default_vt,) + p[1:]
            c = [tok] if tok in avail else [r for r in avail if route_match(p, parse_route(r))]
            if not c:
                print(f"  об'їзд: маршруту «{tok}» у даних немає (див. --list-routes)")
            hit.update(c)
        mask &= df.route.isin(hit).values
        print(f"Об'їзд {detour}: окремо рахуються маршрути {', '.join(sorted(hit)) or '—'}")
    else:
        print(f"Об'їзд {detour}: окремо рахуються всі маршрути")
    return base.where(~mask, base + '_detour')


def day_table(df, overrides):
    rows = []
    expected = list(range(6 * 12, 22 * 12))
    for day, g in df.groupby('day'):
        have = set(g['slot'].unique())
        cov = sum(1 for s in expected if s in have) / len(expected)
        rows.append({'day': day, 'day_type': day_type_of(day, overrides), 'rows': len(g),
                     'vehicles': g['raw_id'].nunique(), 'coverage': round(cov, 3), 'complete': cov >= COMPLETE_SHARE})
    return pd.DataFrame(rows).sort_values('day')


def window_of(hour):
    for name, (a, b) in WINDOWS.items():
        if a <= hour < b:
            return name
    return 'інше'


# ----------------------------- кількість ТЗ ------------------------------
def vehicles_tables(df):
    act = (df.groupby(['route', 'day', 'day_type', 'slot'])['raw_id'].nunique().rename('veh').reset_index())
    pv = act.pivot_table(index=['route', 'day', 'day_type'], columns='slot', values='veh', fill_value=0)
    pv = pv.reindex(columns=range(288), fill_value=0)
    rows = []
    for name, (a, b) in WINDOWS.items():
        cols = list(range(a * 12, b * 12))
        t = pd.DataFrame({'mean': pv[cols].mean(axis=1), 'max': pv[cols].max(axis=1)}).reset_index()
        t['window'] = name
        rows.append(t)
    w = pd.concat(rows)
    win = (w.groupby(['route', 'day_type', 'window'])
             .agg(days=('day', 'nunique'), avg_vehicles=('mean', 'mean'), avg_peak_in_window=('max', 'mean'))
             .round(1).reset_index())
    day_peak = pv.max(axis=1).rename('peak').reset_index()
    peak = (day_peak.groupby(['route', 'day_type'])
            .agg(days=('day', 'nunique'), avg_daily_peak=('peak', 'mean'), max_daily_peak=('peak', 'max'))
            .round(1).reset_index())
    return pv, win, peak


# ----------------------------- відсів нереальних рейсів ------------------
def clean_trips(seg, label='за трасою'):
    n0 = len(seg)
    seg = seg[(seg.total_min >= MIN_TRIP_MIN) & (seg.total_min <= MAX_TRIP_MIN)].copy()
    med = seg.groupby(['route', 'dir']).total_min.transform('median')
    seg = seg[(seg.total_min >= 0.4 * med) & (seg.total_min <= 2.5 * med)].copy()
    print(f"  рейсів {label}: знайдено повних {n0}, після відсіву нереальних {len(seg)}")
    return seg


def trip_tables(trips, win):
    if trips.empty:
        return pd.DataFrame(), pd.DataFrame()
    trips = trips.copy()
    trips['variant'] = trips['variant'].fillna('main') if 'variant' in trips.columns else 'main'
    allday = trips.assign(window='весь день')
    base = pd.concat([trips[trips.window != 'інше'], allday])
    q = lambda p: (lambda s: s.quantile(p))
    tt = (base.groupby(['route', 'dir', 'variant', 'day_type', 'window'])
          .agg(trips=('total_min', 'size'), median_min=('total_min', 'median'), mean_min=('total_min', 'mean'),
               p25_min=('total_min', q(0.25)), p75_min=('total_min', q(0.75)),
               moving_median_min=('moving_min', 'median')).round(1).reset_index())
    cyc = (tt.pivot_table(index=['route', 'day_type', 'window'], columns='dir', values='median_min').reset_index())
    cols = [c for c in cyc.columns if c in (0.0, 1.0)]
    cyc['cycle_min'] = cyc[cols].sum(axis=1, min_count=1).round(1) if cols else np.nan
    cyc['both_directions'] = cyc[cols].notna().all(axis=1) if len(cols) == 2 else False
    cyc = cyc.merge(win.rename(columns={'window': 'window'}), on=['route', 'day_type', 'window'], how='left')
    cyc['interval_est_min'] = (cyc.cycle_min / cyc.avg_vehicles).round(1)
    keep = ['route', 'day_type', 'window', 'cycle_min', 'both_directions', 'avg_vehicles', 'interval_est_min']
    return tt, cyc[keep]


# ----------------------------- звірка за трасою (геометрія) ---------------
class Shape:
    """Траса маршруту: проєкція точок на ламану, прогрес у метрах від початку."""

    def __init__(self, lat, lon):
        self.lat0, self.lon0 = float(np.mean(lat)), float(np.mean(lon))
        self.kx, self.ky = 111320 * np.cos(np.radians(self.lat0)), 110540.0
        x, y = (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky
        keep = np.r_[True, (np.diff(x) != 0) | (np.diff(y) != 0)]
        self.x, self.y = x[keep], y[keep]
        self.dx, self.dy = np.diff(self.x), np.diff(self.y)
        self.seglen = np.hypot(self.dx, self.dy)
        self.cum = np.r_[0, np.cumsum(self.seglen)]
        self.L = float(self.cum[-1])
        self.circular = bool(np.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1]) < 300)
        self.variant, self.vfrom, self.vto, self.sid = 'main', '', '', ''     # варіант траси та період його дії

    def project(self, lat, lon, chunk=400):
        px, py = (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky
        ax, ay = self.x[:-1], self.y[:-1]
        l2 = np.maximum(self.dx ** 2 + self.dy ** 2, 1e-9)
        S, D = np.empty(len(px)), np.empty(len(px))
        for i in range(0, len(px), chunk):
            qx, qy = px[i:i + chunk, None], py[i:i + chunk, None]
            t = np.clip(((qx - ax) * self.dx + (qy - ay) * self.dy) / l2, 0, 1)
            d2 = (qx - (ax + t * self.dx)) ** 2 + (qy - (ay + t * self.dy)) ** 2
            j = d2.argmin(axis=1)
            r = np.arange(len(j))
            S[i:i + chunk] = self.cum[j] + t[r, j] * self.seglen[j]
            D[i:i + chunk] = np.sqrt(d2[r, j])
        return S, D


# ------------------ назви напрямків (кінцеві з trip_headsign) ------------------
def clean_headsign(text):
    """Копія з provider.py, щоб підписи збігалися з тим, що показує карта («Прямує до»)."""
    if not text or str(text).strip().lower() == 'nan':
        return ''
    c = re.sub(r'^\s*\d+\s*[-–—:.]*\s*', '', str(text).strip())
    c = re.sub(r'\s*\(\s*\d+\s*\)\s*$', '', c)
    c = re.sub(r'\s*[-–—]\s*\d+\s*$', '', c)
    return c.strip()


DIR_NAMES = {}      # (маршрут, dir) -> «Дубляни, Школа»


def load_dir_names(static_path, df):
    """Напрямок 0/1 -> кінцева з trip_headsign за назвою маршруту й direction_id зі static (trip_id фіду не використовується)."""
    try:
        if os.path.isdir(static_path):
            files = sorted(glob.glob(os.path.join(static_path, 'static-*.zip')))
            if not files:
                return
            static_path = files[-1]
        z = zipfile.ZipFile(static_path)
        rd = lambda n: pd.read_csv(z.open(n), dtype=str, encoding='utf-8-sig')
        tr, rt = rd('trips.txt'), rd('routes.txt')
        if 'trip_headsign' not in tr:
            print("  ⚠️ у trips.txt немає trip_headsign — підписи лишаються «напрям 0/1»")
            return
        tr['head'] = tr['trip_headsign'].map(clean_headsign)
        tr = tr[tr['head'] != ''].copy()

        # назви: за назвою маршруту і direction_id зі static
        rt['short'] = rt['route_short_name'].str.strip()
        t2 = tr.merge(rt[['route_id', 'short']], on='route_id')
        dirc = pd.to_numeric(t2['direction_id'], errors='coerce') if 'direction_id' in t2 else np.nan
        t2['dir'] = dirc if isinstance(dirc, pd.Series) else np.nan
        t2['dir'] = t2['dir'].fillna(t2.trip_id.str.extract(r'_([01])$')[0].astype(float))
        t2 = t2.dropna(subset=['dir'])
        static_heads = {}
        for (sh, dr), g in t2.groupby(['short', 'dir']):
            static_heads.setdefault(parse_route(sh), {})[float(dr)] = g['head'].mode().iloc[0]
        pairs = df[['route', 'dir']].dropna().drop_duplicates()
        for r, dr in pairs.itertuples(index=False):
            key = (r, float(dr))
            if key in DIR_NAMES:
                continue
            pr = parse_route(r)
            for sp, dd in static_heads.items():
                if route_match(pr, sp) and float(dr) in dd:
                    DIR_NAMES[key] = dd[float(dr)]
                    break
        print(f"  назви напрямків: {len(DIR_NAMES)} із {len(pairs)} пар (маршрут, напрям)")
        miss = sorted({f"{r}/{int(dr)}" for r, dr in pairs.itertuples(index=False) if (r, float(dr)) not in DIR_NAMES})
        if miss:
            print(f"  без назви (лишиться «напрям N»): {', '.join(miss[:40])}" + (" …" if len(miss) > 40 else ""))
    except Exception as e:
        print(f"  ⚠️ не вдалося прочитати назви напрямків: {e}")


def dir_label(route, dr):
    """'→ Дубляни, Школа' або запасний варіант 'напрям 0'."""
    if dr is None or pd.isna(dr):
        return 'напрям ?'
    return '→ ' + DIR_NAMES.get((route, float(dr)), f'напрям {int(dr)}')


def load_static(path):
    """Читає static.zip (або беремо найновіший static-*.zip з папки): траси маршрутів."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, 'static-*.zip')))
        if not files:
            sys.exit(f"У папці {path} немає static-*.zip")
        path = files[-1]
    print(f"  траси з {os.path.basename(path)}")
    z = zipfile.ZipFile(path)
    rd = lambda n: pd.read_csv(z.open(n), dtype=str, encoding='utf-8-sig')
    routes, trips, shapes = rd('routes.txt'), rd('trips.txt'), rd('shapes.txt')
    routes['short'] = routes['route_short_name'].str.strip()
    t = trips.merge(routes[['route_id', 'short']], on='route_id')
    dirc = pd.to_numeric(t['direction_id'], errors='coerce') if 'direction_id' in t else np.nan
    t['dir'] = dirc if isinstance(dirc, pd.Series) else np.nan
    t['dir'] = t['dir'].fillna(t.trip_id.str.extract(r'_([01])$')[0].astype(float))
    cnt = (t.groupby(['short', 'shape_id'])
           .agg(n=('trip_id', 'size'), dir=('dir', lambda s_: s_.mode().iloc[0] if s_.notna().any() else np.nan))
           .reset_index())
    cnt['share'] = cnt.n / cnt.groupby('short').n.transform('sum')
    cnt = cnt[cnt.share >= 0.05]
    shapes['seq'] = shapes['shape_pt_sequence'].astype(float)
    shapes = shapes.sort_values(['shape_id', 'seq'])
    out = {}
    for r in cnt.itertuples():
        g = shapes[shapes.shape_id == r.shape_id]
        if len(g) < 2:
            continue
        S = Shape(g.shape_pt_lat.astype(float).values, g.shape_pt_lon.astype(float).values)
        out.setdefault(r.short, []).append((r.shape_id, r.dir, S))
    return out


def load_routes_dir(path):
    """Власні траси: <маршрут>_line.json — GeoJSON із двома Feature (direction: forward = «Туди» = напрям 0,
    backward = «Назад» = напрям 1), координати [довгота, широта]."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, '*_line.json'))):
        name = os.path.basename(f)[:-len('_line.json')]
        try:
            gj = json.load(open(f, encoding='utf-8'))
        except Exception as e:
            print(f"  {name}: не вдалося прочитати ({e})")
            continue
        for i, ft in enumerate(gj.get('features', [])):
            geom, props = ft.get('geometry', {}), ft.get('properties', {})
            coords = geom.get('coordinates', [])
            if geom.get('type') == 'MultiLineString':
                coords = [c for part in coords for c in part]
            if len(coords) < 2:
                continue
            d = str(props.get('direction', '')).lower()
            sdir = 0.0 if d == 'forward' else 1.0 if d == 'backward' else float(i)
            c = np.array(coords, dtype=float)
            S = Shape(c[:, 1], c[:, 0])
            # необов'язкові властивості Feature: variant, valid_from, valid_to (РРРР-ММ-ДД), напр. «extended», «об'їзд», «великі»
            nsame = sum(1 for x in out.get(name, []) if x[1] == sdir)
            S.variant = str(props.get('variant') or ('main' if nsame == 0 else f'v{nsame + 1}'))
            S.vfrom, S.vto = str(props.get('valid_from') or ''), str(props.get('valid_to') or '')
            S.sid = f"{name}:{d or i}" + ('' if S.variant == 'main' else f':{S.variant}')
            out.setdefault(name, []).append((S.sid, sdir, S))
    print(f"  власні траси: {len(out)} маршрутів із {path}")
    return out


def shape_valid(S, day):
    """Чи діє варіант траси S у цей день (РРРР-ММ-ДД); порожні межі = без обмеження."""
    return (not S.vfrom or day >= S.vfrom) and (not S.vto or day <= S.vto)


def get_shapes(shape_dict, route):
    """Знаходить траси для назви маршруту з даних (А03 ↔ A03, А08а ↔ A08a) незалежно від латиниці/кирилиці."""
    if route in shape_dict:
        return shape_dict[route]
    p = parse_route(route)
    for k, v in shape_dict.items():
        if route_match(parse_route(k), p):
            return v
    return None


def find_traversals(ts, s, spd, L, z=ZONE_M, dirs=None):
    """Наскрізні проїзди траси: ТЗ виходить зі стартової зони (s ≤ z) і доїжджає до кінцевої (s ≥ L − z)
    без розривів у даних і без відкату назад. Час масштабується до повної довжини траси."""
    res, n, a, k, end_thr = [], len(ts), None, 0, L - z
    scale = L / max(L - 2 * z, 1.0)
    while k < n:
        if s[k] <= z:
            a, k = k, k + 1
            continue
        if a is None or ts[k] - ts[a] > GAP_SEC:
            a, k = None, k + 1
            continue
        ta = ts[a] + (ts[k] - ts[a]) * (z - s[a]) / max(s[k] - s[a], 1e-6)
        m, maxs, ok, mov = k, s[k], True, 0.0
        while m < n and s[m] < end_thr:
            if m > k:
                if ts[m] - ts[m - 1] > GAP_SEC or s[m] < maxs - BACK_TOL_M:
                    ok = False
                    break
                if spd[m - 1] >= 3:
                    mov += min(ts[m] - ts[m - 1], 120)
            maxs = max(maxs, s[m])
            m += 1
        if ok and m < n and ts[m] - ts[m - 1] <= GAP_SEC:
            tb = ts[m - 1] + (ts[m] - ts[m - 1]) * (end_thr - s[m - 1]) / max(s[m] - s[m - 1], 1e-6)
            dd = np.nan
            if dirs is not None and np.isfinite(dirs[a:m + 1]).any():
                dd = 1.0 if np.nanmean(dirs[a:m + 1]) > 0.5 else 0.0
            res.append((ta, tb, (tb - ta) * scale, mov * scale, dd))
            a, k = None, m
        else:
            a, k = None, max(k + 1, m)
    return res


DETOUR_MIN_KM = 0.3      # скільки км поза лінією всередині рейсу, щоб вважати його рейсом з об'їздом


def off_route_km(lat, lon, d, ts):
    """Скільки км ТЗ проїхав поза лінією (d > D_MAX_M) між першою й останньою точками."""
    if len(lat) < 2:
        return 0.0
    seg = _hav(lat[:-1], lon[:-1], lat[1:], lon[1:])
    m = (d[1:] > D_MAX_M) & (np.diff(ts) <= 120)
    return float(seg[m].sum()) / 1000


def detour_variant(v):
    return 'об’їзд' if v == 'main' else f'{v}+об’їзд'


def thin_path(lat, lon, step=25.0):
    """Проріджує GPS-слід: залишає точки не ближче step м одна до одної (для кандидатної лінії об'їзду)."""
    keep, last = [0], 0
    for i in range(1, len(lat)):
        if _hav(lat[last], lon[last], lat[i], lon[i]) >= step:
            keep.append(i)
            last = i
    if keep[-1] != len(lat) - 1:
        keep.append(len(lat) - 1)
    return [[round(float(lon[i]), 6), round(float(lat[i]), 6)] for i in keep]


def export_detour_candidates(geo, folder):
    """Для рейсів з об'їздом пише кандидатні лінії (GPS-слід одного рейсу на маршрут/день/напрям) у <folder>/<маршрут>_detour_candidates.json:
    готові Feature з variant/valid_from/valid_to — скопіюйте потрібні до <маршрут>_line.json і за потреби перейменуйте variant та змініть дати."""
    paths = geo.attrs.get('detour_paths') or []
    if not paths:
        return
    os.makedirs(folder, exist_ok=True)
    best = {}
    for p in paths:
        k = (p['route'], p['day'], p['shape_dir'])
        if k not in best or len(p['coords']) > len(best[k]['coords']):
            best[k] = p
    by_route = {}
    for p in best.values():
        by_route.setdefault(p['route'], []).append({
            'type': 'Feature',
            'properties': {'direction': 'forward' if p['shape_dir'] == 0 else 'backward',
                           'name': 'Туди' if p['shape_dir'] == 0 else 'Назад',
                           'variant': f"detour_{p['day']}", 'valid_from': p['day'], 'valid_to': p['day'],
                           'note': f"авто зі GPS: ТЗ {p['raw_id']}, {_hhmm(p['t0'])}–{_hhmm(p['t1'])}, об'їзд ≈{p['km']:.1f} км"},
            'geometry': {'type': 'LineString', 'coordinates': p['coords']}})
    for r, feats in by_route.items():
        with open(os.path.join(folder, f'{r}_detour_candidates.json'), 'w', encoding='utf-8') as f:
            json.dump({'type': 'FeatureCollection', 'name': f'{r}_detour_candidates', 'features': feats}, f, ensure_ascii=False)
    print(f"  Кандидатні лінії об'їздів: {sum(len(v) for v in by_route.values())} (маршрутів: {len(by_route)}) → {folder}")


START_ZONE_M, END_ZONE_M = 150, 120      # зона виїзду / приїзду біля кінців траси, м (параметри --start-zone / --end-zone)


def trip_window(s, on, L, circular=False, from_start=False):
    """Реальний виїзд/приїзд ТЗ вздовж траси довжиною L (s — прогрес у м, on — точка лежить на трасі).
    Повертає (i0, i1, arrived, cov) або None.
      i0 — остання точка, коли ТЗ ще стояв на початковій (момент рушання з кінцевої);
      i1 — перша точка в межах 120 м від кінця траси (момент приїзду; для кільцевої — 350 м);
      cov — яка частка довжини траси охоплена між ними."""
    idx = np.flatnonzero(on)
    if len(idx) < 3:
        return None
    ss = s[idx].astype(float)
    if circular:                       # кінцеві близько одна від одної: хвіст траси, що «склеївся» з початком, віднімаємо
        ss = np.where(ss > L - 200, ss - L, ss)
    if from_start:
        # лише повні проїзди: виїзд зі стартової зони (≤150 м від початку) і приїзд до кінця без повернення назад на старт
        thr_, near, n_ = L - (350 if circular else END_ZONE_M), ss <= START_ZONE_M, len(ss)
        k = 0
        while k < n_:
            if not near[k]:
                k += 1
                continue
            e = k
            while e + 1 < n_ and near[e + 1]:
                e += 1
            m = e + 1
            while m < n_ and ss[m] > START_ZONE_M and ss[m] < thr_:
                m += 1
            if m < n_ and ss[m] >= thr_:
                ref = ss[k:e + 1].min()
                j = k + int(np.flatnonzero(ss[k:e + 1] <= ref + 40)[-1])
                return int(idx[j]), int(idx[m]), True, float((ss[m] - ss[j]) / L)
            k = e + 1
        return None
    ahead = np.flatnonzero(ss >= ss[0] + 150)
    if not len(ahead):
        return None
    k1 = ahead[0]
    ref = ss[:k1 + 1].min()
    j = int(np.flatnonzero(ss[:k1 + 1] <= ref + 40)[-1])
    thr = L - (350 if circular else END_ZONE_M)
    end = np.flatnonzero(ss[j:] >= thr)
    if len(end):
        k2, arrived = j + int(end[0]), True
    else:
        k2, arrived = len(ss) - 1, False
    return int(idx[j]), int(idx[k2]), arrived, float((ss[k2] - ss[j]) / L)


def variant_parts(S, s, d, spd, ts):
    """Складові відповідності руху трасі S: (частка точок на трасі, охоплена частка довжини, частка руху вперед, точок на трасі)."""
    on = d <= D_MAX_M
    if on.sum() < 8:
        return float(on.mean()) if len(on) else 0.0, 0.0, 0.0, int(on.sum())
    cover = (s[on].max() - s[on].min()) / max(S.L, 1.0)
    dt, ds = np.diff(ts), np.diff(s)
    m = on[1:] & on[:-1] & (dt > 0) & (dt <= 120) & (spd[1:] > 5) & (np.abs(ds) < 500)
    fwd = float((ds[m] > 0).mean()) if m.sum() >= 5 else 1.0
    return float(on.mean()), float(min(cover, 1.0)), fwd, int(on.sum())


def variant_score_sd(S, s, d, spd, ts):
    """Наскільки відрізок руху відповідає трасі S: частка точок на трасі × охоплена довжина × частка руху вперед."""
    on_f, cover, fwd, n_on = variant_parts(S, s, d, spd, ts)
    return 0.0 if n_on < 8 else on_f * cover * fwd


def variant_score(S, lat, lon, spd, ts):
    s, d = S.project(lat, lon)
    return variant_score_sd(S, s, d, spd, ts)


def geometry_trips(df, shape_sets, routes_list, min_cov=0.85, min_sec=300):
    """Єдиний метод часу рейсу: від рушання з початкової до приїзду на кінцеву (стоянка на кінцевій не входить).
    Рейс — це проїзд ≥ min_cov довжини траси; решта (неповні, уривки) в статистику не потрапляє.
    Для кожного відрізка руху обирається варіант траси, чинний у цей день (див. variant/valid_from/valid_to у lines).
    shape_sets — список (мітка, {маршрут: [(shape_id, dir, Shape)]}), беруться за порядком пріоритету."""
    rows, det_paths = [], []
    dmap = dict(zip(df.day, df.day_type))
    for r in routes_list:
        label, shp = None, None
        for lb, dct in shape_sets:
            shp = get_shapes(dct, r)
            if shp:
                label = lb
                break
        if not shp:
            print(f"  {r}: траси не знайдено в жодному джерелі — рейси пропущено")
            continue
        sub = df[df.route == r].sort_values(['raw_id', 'ts'])
        if sub.empty:
            continue
        n0, partial, miss = len(rows), 0, []
        for (rid, day), g in sub.groupby(['raw_id', 'day'], sort=False):
            ts, lat, lon = g.ts.values, g.lat.values, g.lon.values
            spd = g.speed.fillna(0).values
            dirs = g.dir.values.astype(float)
            starts = np.flatnonzero(np.r_[True, np.diff(ts) > 600])
            for a_, b_ in zip(starts, np.r_[starts[1:], len(g)]):
                if b_ - a_ < 8:
                    continue
                sl = slice(a_, b_)
                scored, proj = [], {}
                for sid, sdir, S in shp:
                    if not shape_valid(S, day):
                        continue
                    s, d = S.project(lat[sl], lon[sl])
                    proj[sid] = (s, d)
                    scored.append((variant_score_sd(S, s, d, spd[sl], ts[sl]), sid, sdir, S))
                scored.sort(key=lambda z: -z[0])
                used = []
                for ci, (sc, sid, sdir, S) in enumerate(scored):
                    if sc < 0.2:
                        break
                    s, d = proj[sid]
                    on = d <= D_MAX_M
                    pos, n = 0, len(s)
                    while pos < n - 2:
                        w = trip_window(s[pos:], on[pos:], S.L, S.circular, from_start=True)
                        if w is None:
                            if pos == 0 and ci == 0 and on.sum() >= 8 and s[on].max() >= 0.6 * S.L and s[on].min() <= 0.5 * S.L:
                                miss.append(S.L - s[on].max())      # проїхав більшість, але до кінця лінії не дотягнув
                            break
                        i0, i1, arrived, cov = w
                        i0, i1 = i0 + pos, i1 + pos
                        ta, tb = float(ts[a_ + i0]), float(ts[a_ + i1])
                        if arrived and cov >= min_cov and tb - ta >= min_sec:
                            if not any(min(tb, u1) - max(ta, u0) > 0.3 * (tb - ta) for u0, u1 in used):
                                seg = slice(a_ + i0, a_ + i1 + 1)
                                dd = np.nan
                                if np.isfinite(dirs[seg]).any():
                                    dd = 1.0 if np.nanmean(dirs[seg]) > 0.5 else 0.0
                                tt_, sp_ = ts[seg], spd[seg]
                                mov = float(np.minimum(np.diff(tt_), 120)[sp_[:-1] >= 3].sum())
                                dk = off_route_km(lat[seg], lon[seg], d[i0:i1 + 1], tt_)
                                is_det = dk >= DETOUR_MIN_KM
                                if is_det:
                                    det_paths.append({'route': r, 'day': day, 'shape_dir': sdir, 'raw_id': rid, 'km': dk, 't0': ta, 't1': tb,
                                                      'coords': thin_path(lat[seg], lon[seg])})
                                rows.append({'route': r, 'shape_id': sid, 'shape_dir': sdir,
                                             'variant': detour_variant(S.variant) if is_det else S.variant, 'detour_km': round(dk, 2),
                                             'dir_data': dd, 'dir': sdir, 'raw_id': rid, 'start': ta, 'end': tb,
                                             'day': day, 'total_min': (tb - ta) / 60, 'moving_min': mov / 60,
                                             'length_km': S.L / 1000, 'cov': cov})
                                used.append((ta, tb))
                        else:
                            partial += 1
                        pos = max(i1 + 1, pos + 1)
        msg = f"  {r}: траса з «{label}», повних рейсів {len(rows) - n0}, неповних/уривків {partial}"
        if miss:
            msg += f"; не дотягнули до кінця лінії {len(miss)} (бракує в середньому {np.median(miss):.0f} м — якщо багато, скоротіть лінію до реальної кінцевої або збільште --end-zone)"
        if len(rows) == n0:
            msg += "  ← рейсів не знайдено, перевірте лінію/напрямок"
        nd = sum(1 for z in rows[n0:] if z['detour_km'] >= DETOUR_MIN_KM)
        if nd:
            msg += f"; з об'їздом {nd} (≥{DETOUR_MIN_KM} км поза лінією — рахуються окремо)"
        print(msg)
    if not rows:
        return pd.DataFrame()
    g = pd.DataFrame(rows)
    g.attrs['detour_paths'] = det_paths
    # напрям (0/1) рейсу = переважна мітка dir у даних для цієї траси; усі рейси однієї траси мають один напрям,
    # навіть якщо мітка dir у частині рейсів хибна
    dm = g.groupby(['route', 'shape_id']).dir_data.agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else np.nan)
    g['dir'] = [dm.get((r_, s_)) if pd.notna(dm.get((r_, s_))) else sd for r_, s_, sd in zip(g.route, g.shape_id, g.shape_dir)]
    for r_, gr in g.groupby('route'):                  # дві траси різних напрямів не можуть мати однаковий dir — тоді беремо напрям із файлу ліній
        pairs = gr[['shape_id', 'shape_dir', 'dir']].drop_duplicates()
        if pairs.groupby('dir').shape_dir.nunique().gt(1).any():
            g.loc[gr.index, 'dir'] = gr.shape_dir
    loc = pd.to_datetime(g.start, unit='s', utc=True).dt.tz_convert('Europe/Kyiv')
    g['day_type'] = g.day.map(dmap)
    g['start_hour'] = loc.dt.hour
    g['window'] = g.start_hour.map(window_of)
    return g


# ----------------------------- швидкості ---------------------------------
def speed_tables(df):
    mv = df[df.speed >= 3].copy()
    mv['window'] = mv['hour'].map(window_of)
    by_route = (mv[mv.window != 'інше'].groupby(['route', 'day_type', 'window']).speed
                .agg(avg_speed_kmh='mean', points='size').round(1).reset_index())
    by_hour = (mv.groupby(['vtype', 'day_type', 'hour']).speed.agg(avg_speed_kmh='mean', points='size')
               .round(1).reset_index())
    return by_route, by_hour


# ----------------------------- план vs факт ------------------------------
def _cell(x):
    """Порожня клітинка CSV (NaN) -> ''."""
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else str(x).strip()


def plan_by_day(plan_path, pv, uniq=None):
    """Виконання плану випуску по днях: для кожного рядка плану і кожного дня — план, фактичний пік одночасних ТЗ
    і кількість різних ТЗ за добу (uniq — Series (route, day) -> n). Повертає DataFrame."""
    plan = pd.read_csv(plan_path, dtype=str, sep=None, engine='python', encoding='utf-8-sig')
    peak = pv.max(axis=1).rename('peak').reset_index()
    out = []
    for _, r in plan.iterrows():
        vt = VTYPE_ALIASES.get(str(r.get('type', '')).strip().lower())
        pr = parse_route(r['route'], vt)
        dtp = DAY_ALIASES.get(str(r['day_type']).strip().lower())
        if pr[1] is None or not dtp or not str(r['planned']).strip().isdigit():
            continue
        planned = int(r['planned'])
        vf = _cell(r.get('valid_from')) or '1900-01-01'
        vto = _cell(r.get('valid_to')) or '2999-12-31'
        sub = peak[peak.route.map(lambda x: route_match(pr, parse_route(x)))]
        sub = sub[sub.day.map(lambda d: vf <= d <= vto) & (sub.day_type == dtp)]
        for day, g in sub.groupby('day'):
            names = sorted(g.route.unique())
            u = int(sum(uniq.get((n_, day), 0) for n_ in names)) if uniq is not None else None
            out.append({'route': '/'.join(names), 'names': names, 'day': day, 'day_type': dtp, 'planned': planned,
                        'peak': int(g.peak.sum()), 'unique': u})
    return pd.DataFrame(out)


def plan_vs_fact(plan_path, pv, days_df):
    plan = pd.read_csv(plan_path, dtype=str, sep=None, engine='python', encoding='utf-8-sig')
    out = []
    peak = pv.max(axis=1).rename('peak').reset_index()
    for _, r in plan.iterrows():
        vt = VTYPE_ALIASES.get(str(r.get('type', '')).strip().lower())
        pr = parse_route(r['route'], vt)
        dtp = DAY_ALIASES.get(str(r['day_type']).strip().lower())
        if pr[1] is None or not dtp or not str(r['planned']).strip().isdigit():
            continue
        planned = int(r['planned'])
        vf = _cell(r.get('valid_from')) or '1900-01-01'
        vto = _cell(r.get('valid_to')) or '2999-12-31'
        sub = peak[peak.route.map(lambda x: route_match(pr, parse_route(x)))]
        sub = sub[sub.day.map(lambda d: vf <= d <= vto)]
        sub = sub[sub.day_type == dtp]
        if sub.empty:
            out.append({'route': r['route'], 'type': vt or '', 'day_type': dtp, 'planned': planned, 'days': 0,
                        'avg_peak': 0.0, 'pct_of_plan': 0.0, 'note': 'у даних таких днів/маршруту немає'})
            continue
        names = sorted(sub.route.unique())
        kinds = {parse_route(x)[0] for x in names}
        per_day = sub.groupby('day').peak.sum()
        note = ''
        if len(kinds) > 1 and pr[0] is None:
            note = f"неоднозначно: {', '.join(names)} — додайте type"
        out.append({'route': '/'.join(names), 'type': vt or '', 'day_type': dtp, 'planned': planned,
                    'days': len(per_day), 'avg_peak': round(per_day.mean(), 1),
                    'pct_of_plan': round(100 * per_day.mean() / planned, 0) if planned else np.nan, 'note': note})
    return pd.DataFrame(out)


# ----------------------------- графіки -----------------------------------
def make_charts(out, pv, tt, speeds_hour, routes, plan_df, cmp=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не встановлено — графіки пропущено (pip install matplotlib)")
        return
    x = np.arange(288) / 12
    # 1) профіль кількості ТЗ по маршрутах
    for r in routes:
        sub = pv[pv.index.get_level_values('route') == r]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        for dtp, g in sub.groupby(level='day_type'):
            ax.plot(x, g.mean(axis=0).values, label=f"{DAY_TYPE_UA.get(dtp, dtp)} (днів: {len(g)})")
        ax.set_title(f"Маршрут {r}: скільки ТЗ на лінії протягом доби (середнє)")
        ax.set_xlabel('година доби'); ax.set_ylabel('ТЗ одночасно'); ax.set_xlim(4, 24); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, f"profile_{r}.png"), dpi=130); plt.close(fig)
    # 2) час рейсу за частиною дня
    if not tt.empty:
        for r in routes:
            sub = tt[(tt.route == r) & (tt.window != 'весь день')]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 4))
            order = list(WINDOWS.keys())
            if 'variant' not in sub.columns:
                sub = sub.assign(variant='main')
            sub = sub.assign(variant=sub.variant.fillna('main'))
            combos = sub.groupby(['day_type', 'dir', 'variant']).ngroups
            width = 0.8 / max(combos, 1)
            for i, ((dtp, dr, var), g) in enumerate(sub.groupby(['day_type', 'dir', 'variant'])):
                g = g.set_index('window').reindex(order)
                pos = np.arange(len(order)) + i * width
                err = [(g.median_min - g.p25_min).fillna(0).values, (g.p75_min - g.median_min).fillna(0).values]
                bars = ax.bar(pos, g.median_min.fillna(0).values, width, yerr=err, capsize=3,
                       label=f"{DAY_TYPE_UA.get(dtp, dtp)}, {dir_label(r, dr)}{'' if var == 'main' else ' [' + var + ']'} (n={int(g.trips.sum())})")
                for bar_, nn_ in zip(bars, g.trips.fillna(0).values):
                    if nn_ < 3:
                        bar_.set_alpha(0.3)             # менше 3 рейсів — медіана ненадійна
            ax.set_xticks(np.arange(len(order)) + width * (combos - 1) / 2); ax.set_xticklabels(order)
            ax.set_ylabel('хв (медіана, лінія — міжквартильний розмах)')
            meth = sub.method.iloc[0] if 'method' in sub.columns else ''
            ax.set_title(f"Маршрут {r}: час рейсу (" + ('від виїзду з кінцевої до приїзду') + "; прозорі стовпці — менше 3 рейсів", fontsize=10)
            ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(out, f"trip_time_{r}.png"), dpi=130); plt.close(fig)
    # 3) швидкість по годинах за видами
    if not speeds_hour.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        for (vt, dtp), g in speeds_hour.groupby(['vtype', 'day_type']):
            ax.plot(g.hour, g.avg_speed_kmh, marker='o', ms=3, label=f"{vt}, {DAY_TYPE_UA.get(dtp, dtp)}")
        ax.set_title('Середня швидкість руху (≥ 3 км/год) по годинах'); ax.set_xlabel('година'); ax.set_ylabel('км/год')
        ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, 'speed_by_hour.png'), dpi=130); plt.close(fig)
    # 4) план vs факт
    if plan_df is not None and not plan_df.empty:
        p = plan_df[plan_df.days > 0].sort_values('pct_of_plan').head(30)
        if not p.empty:
            fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(p))))
            labels = [f"{r.route} ({DAY_TYPE_UA.get(r.day_type, r.day_type)})" for r in p.itertuples()]
            ax.barh(labels, p.pct_of_plan, color=['#c0392b' if v < 70 else '#e67e22' if v < 90 else '#27ae60'
                                                  for v in p.pct_of_plan])
            ax.axvline(100, color='k', lw=1); ax.set_xlabel('піковий випуск, % від плану')
            ax.set_title('Виконання планового випуску (30 найгірших маршрутів)')
            fig.tight_layout(); fig.savefig(os.path.join(out, 'plan_vs_fact.png'), dpi=130); plt.close(fig)


# ----------------------------- підсумок ----------------------------------
def write_report(out, days_df, used, routes, win, peak, tt, cyc, plan_df, cmp=None, tt_geo=None):
    L = ["# Підсумок аналізу", ""]
    L.append(f"Дні в аналізі: {', '.join(used)} (усього знайдено днів: {len(days_df)}).")
    if len(used) == 1:
        L.append("⚠️ Це один день: числа нижче — зріз цього дня, а не середні; будні/вихідні не порівнюються.")
    skipped = days_df[~days_df.day.isin(used)]
    if len(skipped):
        L.append(f"Пропущено неповні дні: {', '.join(skipped.day)} (покриття < {int(COMPLETE_SHARE * 100)}%).")
    L.append("")
    for r in routes:
        L.append(f"## Маршрут {r}")
        for dtp in sorted(win[win.route == r].day_type.unique()):
            L.append(f"**{DAY_TYPE_UA.get(dtp, dtp)}**")
            for w in list(WINDOWS.keys()):
                a = win[(win.route == r) & (win.day_type == dtp) & (win.window == w)]
                t = tt[(tt.route == r) & (tt.day_type == dtp) & (tt.window == w)] if not tt.empty else tt
                c = cyc[(cyc.route == r) & (cyc.day_type == dtp) & (cyc.window == w)] if not cyc.empty else cyc
                s = f"- {w}: "
                s += f"ТЗ у середньому {a.avg_vehicles.iloc[0]}" if len(a) else "немає даних про ТЗ"
                for t_ in t.itertuples():
                    s += (f"; {dir_label(r, t_.dir)}: рейс {t_.median_min} хв "
                          f"(міжквартильно {t_.p25_min}–{t_.p75_min}, у русі ≈{t_.moving_median_min}, n={t_.trips})")
                if len(c) and pd.notna(c.interval_est_min.iloc[0]):
                    s += f"; коло ≈{c.cycle_min.iloc[0]} хв, інтервал ≈{c.interval_est_min.iloc[0]} хв"
                L.append(s)
            L.append("")
    if tt_geo is not None and not tt_geo.empty:
        L.append("## Час рейсу за трасою (геометрія)")
        for r in routes:
            for c in tt_geo[tt_geo.route == r].sort_values(['day_type', 'dir', 'window']).itertuples():
                L.append(f"- {r}, {DAY_TYPE_UA.get(c.day_type, c.day_type)}, {dir_label(r, c.dir)}, "
                         f"{c.window}: {c.median_min} хв (міжквартильно {c.p25_min}–{c.p75_min}, n={c.trips})")
        L.append("")
    if plan_df is not None and not plan_df.empty:
        L.append("## План vs факт (піковий одночасний випуск)")
        for r in plan_df.itertuples():
            L.append(f"- {r.route} ({DAY_TYPE_UA.get(r.day_type, r.day_type)}): план {r.planned}, факт {r.avg_peak} "
                     f"({r.pct_of_plan:.0f}%), днів {r.days} {r.note}")
    with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('\n'.join(L[:REPORT_CONSOLE_LINES]))
    if len(L) > REPORT_CONSOLE_LINES:
        print(f"\n... (ще {len(L) - REPORT_CONSOLE_LINES} рядків, повний звіт: {os.path.join(out, 'report.md')})")


# ----------------------------- якість одного дня --------------------------
def _hav(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _hhmm(ts):
    return pd.to_datetime(ts, unit='s', utc=True).tz_convert('Europe/Kyiv').strftime('%H:%M')


def day_quality(d, day, day_type, out):
    """Діагностика одного дня: чи нормально працював збір і що за дані. Працює і на неповному дні."""
    L, flags, info = [], [], []
    d = d.sort_values(['raw_id', 'ts']).copy()
    n, veh = len(d), d.raw_id.nunique()
    L.append(f"# Якість даних за {day} ({DAY_TYPE_UA.get(day_type, day_type)})\n")
    L.append(f"Рядків: **{n}**, унікальних ТЗ (трекерів): **{veh}**, маршрутів: **{d.route.nunique()}**. "
             f"Перша точка {_hhmm(d.ts.min())}, остання {_hhmm(d.ts.max())} (за Києвом).\n")

    # 1) покриття й прогалини всього збору
    exp = set(range(6 * 12, 22 * 12))
    cov = len(exp & set(d.slot.unique())) / len(exp)
    L.append(f"## 1. Покриття збору\nІнтервалів по 5 хв між 06:00 і 22:00 з даними: **{cov * 100:.0f}%**.")
    u = np.sort(d.ts.unique())
    def _min_of_day(t):
        x = pd.to_datetime(t, unit='s', utc=True).tz_convert('Europe/Kyiv')
        return x.hour * 60 + x.minute
    gaps = [(a, b) for a, b in zip(u[:-1], u[1:])
            if b - a > 180 and _min_of_day(b) > 4 * 60 + 30]       # нічну паузу колектора (00:00-04:00) не рахуємо
    long_gaps = [(a, b) for a, b in gaps if b - a >= 600]
    if gaps:
        L.append("Прогалини понад 3 хв без жодної точки від жодного ТЗ (збій збору, фіду або перезапуск):")
        for a, b in sorted(gaps, key=lambda x: x[0] - x[1])[:10]:
            L.append(f"- {_hhmm(a)} → {_hhmm(b)} ({(b - a) / 60:.0f} хв)")
    else:
        L.append("Прогалин понад 3 хв немає.")
    L.append("")
    if cov < COMPLETE_SHARE:
        flags.append(f"⚠️ покриття {cov * 100:.0f}% < {int(COMPLETE_SHARE * 100)}%: день неповний, частини доби недостовірні")
    if long_gaps:
        flags.append(f"⚠️ є {len(long_gaps)} прогалин від 10 хв")

    # 2) частота точок
    per = d.groupby('raw_id').ts.agg(['size', lambda s_: np.median(np.diff(s_)) if len(s_) > 1 else np.nan])
    per.columns = ['points', 'median_dt']
    med_dt = per.median_dt.median()
    L.append("## 2. Частота точок")
    L.append(f"Медіанний інтервал між точками одного ТЗ: **{med_dt:.0f} с** (очікувано ~33 с при записі раз на 30 с). "
             f"Точок на ТЗ: медіана {per.points.median():.0f}, мінімум {per.points.min()}, максимум {per.points.max()}. "
             f"ТЗ з менше ніж 10 точок: {(per.points < 10).sum()} (короткі виходи або «привиди»).\n")
    if med_dt > 60:
        flags.append(f"⚠️ медіанний інтервал {med_dt:.0f} с — точки надходять рідше, ніж очікується")

    # 3) скільки ТЗ було на лінії по годинах
    L.append("## 3. ТЗ на лінії по годинах (унікальних ТЗ за годину)")
    hv = d.pivot_table(index='hour', columns='vtype', values='raw_id', aggfunc='nunique', fill_value=0)
    hv['усього'] = d.groupby('hour').raw_id.nunique()
    L.append("```\n" + hv.to_string() + "\n```")
    peak5 = d.groupby('slot').raw_id.nunique()
    L.append(f"Пік одночасно (5-хв інтервал): **{peak5.max()}** ТЗ о {int(peak5.idxmax() // 12):02d}:{int(peak5.idxmax() % 12 * 5):02d}.\n")

    # 4) відомі борти й дублі
    unk = d.plate.isna() | (d.plate.astype(str) == '')
    L.append("## 4. Борти (зіставлення трекер → ТЗ)")
    L.append(f"Рядків без визначеного борту: **{unk.mean() * 100:.1f}%**.")
    kn = d[~unk].copy()
    kn['minute'] = kn.ts // 60
    dup = kn.groupby(['plate', 'minute']).raw_id.nunique()
    dup_plates = dup[dup > 1].reset_index().plate.nunique()
    L.append(f"Бортів, що в ту саму хвилину значаться на двох різних трекерах: **{dup_plates}** (ознака помилки mapping).\n")
    if unk.mean() > 0.10:
        flags.append(f"⚠️ {unk.mean() * 100:.0f}% точок без борту")
    if dup_plates:
        flags.append(f"⚠️ {dup_plates} бортів на двох трекерах одночасно — перевірте vehicle_mapping")

    # 5) рух: швидкості, стрибки, поза трасою
    L.append("## 5. Рух і геометрія")
    sp = d.speed
    prev_same = d.raw_id.eq(d.raw_id.shift())
    dt = (d.ts - d.ts.shift()).where(prev_same)
    dist = _hav(d.lat.shift(), d.lon.shift(), d.lat, d.lon)
    imp = (dist / dt * 3.6).where(prev_same & (dt > 0) & (dt <= 120))
    jumps = int((imp > 120).sum())
    L.append(f"Швидкість, км/год: медіана {sp.median():.0f}, 95-й перцентиль {sp.quantile(.95):.0f}, максимум {sp.max():.0f}. "
             f"Стоять (<3 км/год): {(sp < 3).mean() * 100:.0f}% точок. Швидкість понад 80 км/год: {(sp > 80).sum()} точок.")
    L.append(f"Нереальні стрибки координат (>120 км/год між сусідніми точками): **{jumps}** "
             f"({jumps / max(imp.notna().sum(), 1) * 100:.2f}% пар).")
    if 'dist_m' in d:
        L.append(f"Поза трасою (>100 м): {(d.off_route == 1).mean() * 100:.1f}% точок. "
                 f"Точок без траси (рейс не відомий статичному GTFS): {d.dist_m.isna().mean() * 100:.1f}%.")
    L.append("")
    if jumps / max(imp.notna().sum(), 1) > 0.005:
        flags.append("⚠️ багато стрибків координат (>0,5% пар)")

    # 6) напрямок
    L.append("## 6. Напрямок у даних")
    L.append(f"Рядків без напрямку в даних: **{d.dir_raw.isna().mean() * 100:.0f}%**. "
             "Напрямок рейсів у звітах визначається за лінією маршруту, а не за міткою в даних.\n")

    # 7) маршрути
    L.append("## 7. Маршрути (топ-10 за кількістю ТЗ)")
    rt = d.groupby('route').raw_id.nunique().sort_values(ascending=False).head(10)
    L.append("```\n" + rt.to_string() + "\n```\n")

    # підсумок
    L.insert(1, "## Підсумок\n" + ("\n".join(f"- {f}" for f in flags) if flags else "- ✅ критичних проблем з даними не виявлено") + "\n")
    path = os.path.join(out, f"quality_{day}.md")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        act = d.groupby(['slot', 'vtype']).raw_id.nunique().unstack(fill_value=0).reindex(range(288), fill_value=0)
        for col in act.columns:
            ax[0].plot(act.index / 12, act[col], label=col)
        ax[0].set_ylabel('ТЗ на лінії'); ax[0].set_title(f"{day}: ТЗ на лінії (унікальних за 5 хв)"); ax[0].legend(); ax[0].grid(alpha=.3)
        dens = d.groupby('slot').size().reindex(range(288), fill_value=0)
        ax[1].bar(dens.index / 12, dens.values, width=1 / 12, color='#2c7fb8')
        ax[1].set_ylabel('точок за 5 хв'); ax[1].set_xlabel('година'); ax[1].set_xlim(4, 24); ax[1].grid(alpha=.3)
        ax[1].set_title('Щільність даних (провали = прогалини збору)')
        fig.tight_layout(); fig.savefig(os.path.join(out, f"quality_{day}.png"), dpi=130); plt.close(fig)
    except ImportError:
        pass
    return flags, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='.', help='папка з positions-*.csv.gz')
    ap.add_argument('--out', default='report')
    ap.add_argument('--route', default='', help='маршрути через кому: А03,Т01 (можна «3» або «3а»); '
                                                '«all» — усі маршрути; порожньо — топ-3 за випуском')
    ap.add_argument('--charts-top', type=int, default=0,
                    help='для скількох найбільших маршрутів малювати окремі графіки (0 = для всіх, за замовчуванням)')
    ap.add_argument('--plan', default='', help='CSV з плановим випуском (як для import-plan)')
    ap.add_argument('--day-types', default='', help='CSV day,type для свят і особливих днів')
    ap.add_argument('--start-zone', type=float, default=150, help='зона біля початку траси, звідки рахується виїзд, м')
    ap.add_argument('--end-zone', type=float, default=120, help='зона біля кінця траси, де фіксується приїзд, м (збільште, якщо кінцева стоїть осторонь лінії)')
    ap.add_argument('--detour', default='', help="період об'їзду РРРР-ММ-ДД:РРРР-ММ-ДД (кілька — через кому); дані за нього рахуються окремо")
    ap.add_argument('--detour-routes', default='', help="маршрути, яких стосується об'їзд, через кому (порожньо = усі); номери без літери — автобуси")
    ap.add_argument('--days', default='', help='обмежити днями через кому: 2026-11-10,2026-11-11')
    ap.add_argument('--static', default='', help='static.zip (або папка з static-*.zip) — для звірки часу рейсу за трасою')
    ap.add_argument('--routes-dir', default='', help='папка з власними трасами <маршрут>_line.json (напр. ...\\lviv-transport-web\\public\\routes)')
    ap.add_argument('--compare', action='store_true', help='застарілий параметр, ігнорується: час рейсу завжди рахується за трасою (потрібен --static або --routes-dir)')
    ap.add_argument('--geo-all', action='store_true', help='геометрія для всіх маршрутів (повільно)')
    ap.add_argument('--include-partial', action='store_true', help='брати й неповні дні')
    ap.add_argument('--list-routes', action='store_true')
    a = ap.parse_args()

    only_days = {d.strip() for d in a.days.split(',') if d.strip()}
    print("Завантаження:")
    df = load_positions(a.data, only_days)
    df['dir_raw'] = df['dir']
    overrides = load_day_overrides(a.day_types)
    df['day_type'] = assign_day_types(df, overrides, a.detour, a.detour_routes)
    globals()['START_ZONE_M'], globals()['END_ZONE_M'] = a.start_zone, a.end_zone
    days_df = day_table(df, overrides)

    if a.list_routes:
        g = df.groupby('route').agg(vehicles=('raw_id', 'nunique'), vtype=('vtype', lambda s: s.mode().iloc[0]))
        print(g.sort_index().to_string())
        return

    os.makedirs(a.out, exist_ok=True)
    days_df.to_csv(os.path.join(a.out, 'days.csv'), index=False)
    print("\nДні:\n" + days_df.to_string(index=False))
    used = list(days_df.day) if a.include_partial else list(days_df[days_df.complete].day)
    partial_mode = a.include_partial
    if not used:
        used, partial_mode = list(days_df.day), True
        print("\n⚠️ Немає повних днів (покриття ≥ 95%) — аналізую те, що є. Частини доби (ранок/обід/вечір) і середні "
              "значення будуть недостовірні; корисна насамперед діагностика якості (quality_*.md).")
    df = df[df.day.isin(used)].copy()

    pv, win, peak = vehicles_tables(df)
    for dd in used:
        flags, qp = day_quality(df[df.day == dd], dd, day_type_of(dd, overrides), a.out)
        print(f"\nЯкість {dd}: " + ("; ".join(flags) if flags else "критичних проблем не виявлено") + f"  → {qp}")
    win.to_csv(os.path.join(a.out, 'vehicles_by_window.csv'), index=False)
    peak.to_csv(os.path.join(a.out, 'peak_by_route.csv'), index=False)

    if a.static:
        load_dir_names(a.static, df)
    available = sorted(df.route.unique())
    by_size = list(peak.groupby('route').avg_daily_peak.max().sort_values(ascending=False).index)
    if a.route.strip().lower() in ('all', 'всі', '*'):
        routes = by_size                      # усі маршрути, від найбільших до найменших
    elif a.route:
        routes = resolve_routes(a.route.split(','), available)
    else:
        routes = by_size[:3]
    print(f"\nМаршрутів в аналізі: {len(routes)}")

    tt, cyc = pd.DataFrame(), pd.DataFrame()      # час рейсу рахується лише за трасою (див. geometry_trips)

    tt_geo, cmp, cyc_geo = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not (a.static or a.routes_dir):
        print("\nУВАГА: час рейсу рахується лише за трасою — вкажіть --static і/або --routes-dir (lines).")
    if a.static or a.routes_dir:
        print("\nРейси за трасою:")
        shape_sets = []
        if a.routes_dir:
            shape_sets.append(('власні траси', load_routes_dir(a.routes_dir)))
        if a.static:
            shape_sets.append(('GTFS static', load_static(a.static)))
        geo = geometry_trips(df, shape_sets, available if a.geo_all else routes)
        if not geo.empty:
            export_detour_candidates(geo, os.path.join(a.out, 'detour_lines'))
            geo = clean_trips(geo, 'за трасою')
            geo.to_csv(os.path.join(a.out, 'trips_geometry.csv'), index=False)      # усі повні рейси — для перевірки
            tt_geo, cyc_geo = trip_tables(geo, win)
            tt_geo.to_csv(os.path.join(a.out, 'trip_times.csv'), index=False)
            cyc_geo.to_csv(os.path.join(a.out, 'cycle_interval.csv'), index=False)
        else:
            print("  наскрізних проїздів за трасою не знайдено")

    by_route, by_hour = speed_tables(df)
    by_route.to_csv(os.path.join(a.out, 'speeds_by_route.csv'), index=False)
    by_hour.to_csv(os.path.join(a.out, 'speeds_by_hour.csv'), index=False)

    plan_df = None
    if a.plan:
        plan_df = plan_vs_fact(a.plan, pv, days_df)
        plan_df.to_csv(os.path.join(a.out, 'plan_vs_fact.csv'), index=False)

    # основний час рейсу — за трасою (від виїзду з кінцевої до приїзду, без стоянки)
    tt_main = tt_geo.assign(method='траса') if not tt_geo.empty else pd.DataFrame()
    cyc_main = cyc_geo
    chart_routes = routes[:a.charts_top] if a.charts_top > 0 else routes
    print(f"Малюю графіки для {len(chart_routes)} маршрутів...")
    make_charts(a.out, pv, tt_main, by_hour, chart_routes, plan_df, cmp)
    print()
    write_report(a.out, days_df, used, routes, win, peak, tt_main, cyc_main, plan_df, cmp, tt_geo)
    print(f"\nГотово. Файли та графіки: {os.path.abspath(a.out)}")


if __name__ == '__main__':
    main()
