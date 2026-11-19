#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_site.py — готує дані для вебдашборду (папка site/), використовуючи ваші analyze1.py і speed_map.py.

    python build_site.py --data D:\\transit-data --static D:\\static --lines D:\\lines --out site

Лежить поруч з analyze1.py та speed_map.py. Результат: site/index.html (копіюється з index.html поруч),
site/index.json, site/stops.json, site/data/rNNN.json (по одному файлу на маршрут).
Далі site/ можна відкрити через `python -m http.server -d site` або викласти на GitHub Pages / Render Static.
"""
import subprocess
import argparse
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

import analyze1 as A
import speed_map as SM

WINS = {'всі': (6, 22), 'ранок': (6, 9), 'обід': (11, 14), 'вечір': (16, 19)}
DT_UA = A.DAY_TYPE_UA


def dump(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))


def line_of(S):
    lat = S.lat0 + S.y / S.ky
    lon = S.lon0 + S.x / S.kx
    return [[round(float(a), 5), round(float(b), 5), int(c)] for a, b, c in zip(lat, lon, S.cum)]


def point_at(S, s):
    j = int(np.clip(np.searchsorted(S.cum, s, 'right') - 1, 0, len(S.seglen) - 1))
    t = (s - S.cum[j]) / max(S.seglen[j], 1e-9)
    return (S.lat0 + (S.y[j] + t * S.dy[j]) / S.ky, S.lon0 + (S.x[j] + t * S.dx[j]) / S.kx)


def speed_bins(S, s, d, v, hour, dtype, dtypes, a):
    """Середня швидкість по ділянках траси для кожної комбінації (тип дня × частина доби)."""
    ok = (d <= a.dmax) & (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M) & np.isfinite(v)
    nb = int(math.ceil(S.L / a.bin_m))
    out = {}
    for dt in dtypes + ['all']:
        m_dt = ok if dt == 'all' else ok & (dtype == dt)
        for wn, (lo, hi) in WINS.items():
            m = m_dt & (hour >= lo) & (hour < hi)
            if not m.any():
                continue
            b = np.minimum((s[m] // a.bin_m).astype(int), nb - 1)
            cnt = np.bincount(b, minlength=nb)
            mean = np.bincount(b, weights=v[m], minlength=nb) / np.maximum(cnt, 1)
            out[f'{dt}|{wn}'] = [round(float(mean[i]), 1) if cnt[i] >= a.min_n else None for i in range(nb)]
    return out


def timeline(S, s, d, v, sod, dtype, dtypes, a):
    """Середня швидкість по 10-хв слотах 06:00–22:00 (для графіка)."""
    ok = (d <= a.dmax) & (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M) & np.isfinite(v) & (sod >= 6 * 3600) & (sod < 22 * 3600)
    out = {}
    for dt in dtypes + ['all']:
        m = ok if dt == 'all' else ok & (dtype == dt)
        if not m.any():
            continue
        sl = ((sod[m] - 6 * 3600) // 600).astype(int)
        cnt = np.bincount(sl, minlength=96)[:96]
        mean = np.bincount(sl, weights=v[m], minlength=96)[:96] / np.maximum(cnt, 1)
        out[dt] = [round(float(mean[i]), 1) if cnt[i] >= 5 else None for i in range(96)]
    return out


def replay_tracks(g, s, d, sod, v, day):
    """Траєкторії ТЗ вздовж траси: [[сек_доби, метр_траси, швидкість], ...]; розриваємо на пропусках і відкатах."""
    res = {}
    ok = (d <= A.D_MAX_M) & (sod >= 6 * 3600) & (sod < 22 * 3600)
    for dd in np.unique(day):
        tracks = []
        mday = ok & (day == dd)
        if not mday.any():
            continue
        rid = g.raw_id.values[mday]
        ts, ss, ssod, vv = g.ts.values[mday], s[mday], sod[mday], v[mday]
        cuts = np.flatnonzero(rid[1:] != rid[:-1]) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(rid)]):
            cur, last_t = [], -1e9
            for k in range(lo, hi):
                if cur and (ts[k] - ts[k - 1] > 120 or ss[k] < cur[-1][1] - 500):
                    if len(cur) >= 3:
                        tracks.append({'v': rid[lo], 'p': cur})
                    cur, last_t = [], -1e9
                if ts[k] - last_t >= 20:
                    cur.append([int(ssod[k]), int(ss[k]), int(vv[k]) if np.isfinite(vv[k]) else 0])
                    last_t = ts[k]
            if len(cur) >= 3:
                tracks.append({'v': rid[lo], 'p': cur})
        res[str(dd)] = tracks
    return res


def dwell_points(S, g, s, d, a):
    """Стоянки довше --dwell-sec секунд поза кінцевими зонами; агрегуємо по ділянках."""
    m = (d <= a.dmax) & (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M)
    if not m.any():
        return []
    rid, ts = g.raw_id.values[m], g.ts.values[m]
    sp, ss = g.speed.values[m], s[m]
    lat, lon = g.lat.values[m], g.lon.values[m]
    slow = sp < 3
    prev_slow = np.r_[False, slow[:-1]]
    new = slow & (~prev_slow | np.r_[True, rid[1:] != rid[:-1]] | np.r_[True, np.diff(ts) > 120])
    run = np.cumsum(new) * slow
    rows = {}
    for r in np.unique(run[run > 0]):
        idx = np.flatnonzero(run == r)
        dur = ts[idx[-1]] - ts[idx[0]] + 20
        if dur >= a.dwell_sec:
            b = int(ss[idx[0]] // a.bin_m)
            rows.setdefault(b, []).append((dur, lat[idx[0]], lon[idx[0]]))
    out = []
    for b, lst in rows.items():
        if len(lst) >= a.dwell_min_count:
            out.append([b * a.bin_m, round(float(np.mean([x[1] for x in lst])), 5), round(float(np.mean([x[2] for x in lst])), 5),
                        len(lst), round(float(np.median([x[0] for x in lst])) / 60, 1)])
    out.sort(key=lambda x: (-x[3], -x[4]))
    return out[:12]


def smooth_dir(d, min_pts=5):
    """Напрям зі «миготінням» прибираємо: нова ділянка напряму має тривати щонайменше min_pts точок."""
    d = pd.Series(d).ffill().bfill().values.astype(float)
    if len(d) == 0 or np.isnan(d).all():
        return np.full(len(d), np.nan)
    out, i = d.copy(), 0
    runs = []
    while i < len(d):
        j = i
        while j < len(d) and d[j] == d[i]:
            j += 1
        runs.append([i, j, d[i]])
        i = j
    for k in range(1, len(runs)):
        if runs[k][1] - runs[k][0] < min_pts:
            runs[k][2] = runs[k - 1][2]
    for a_, b_, v_ in runs:
        out[a_:b_] = v_
    return out


def hav_m(la, lo):
    if len(la) < 2:
        return 0.0
    la, lo = np.radians(la), np.radians(lo)
    h = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return float(np.sum(2 * 6371000 * np.arcsin(np.sqrt(h))))


def make_seg(x, s, d, r_, dd, S, a, vid=None):
    """Один відрізок руху ТЗ -> dict. ty='trip' (повний рейс) або 'part' (короткий/неповний відрізок)."""
    sod = x.sod.values
    first, last = int(sod[0]), int(sod[-1])
    ty, cov, i0, i1 = 'part', None, 0, len(x) - 1
    if S is not None:
        w = A.trip_window(s, d <= A.D_MAX_M, S.L, S.circular)
        if w:
            i0, i1, arrived, cov = w
            if cov >= a.min_cov and sod[i1] - sod[i0] >= 300:
                ty = 'trip'
    if i1 - i0 < 1:
        return None
    dk = A.off_route_km(x.lat.values[i0:i1 + 1], x.lon.values[i0:i1 + 1], d[i0:i1 + 1], x.ts.values[i0:i1 + 1]) if S is not None else 0.0
    x, s = x.iloc[i0:i1 + 1], (s[i0:i1 + 1] if s is not None else None)
    sp = x.speed.fillna(0).values
    ts = x.ts.values
    pts, last_ts = [], -1e9
    for t_, so, la, lo_, v_ in zip(ts, x.sod.values, x.lat.values, x.lon.values, sp):
        if t_ - last_ts >= 15:
            pts.append([int(so), round(float(la), 5), round(float(lo_), 5), int(v_)])
            last_ts = t_
    # простої >= dwell-sec; кінцеві зони не рахуємо
    ok = np.ones(len(x), bool) if S is None else (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M)
    slow = (sp < 3) & ok
    dw, run0 = 0, None
    for k in range(len(x)):
        if slow[k]:
            run0 = k if run0 is None else run0
        else:
            if run0 is not None and ts[k - 1] - ts[run0] + 20 >= a.dwell_sec:
                dw += 1
            run0 = None
    if run0 is not None and ts[-1] - ts[run0] + 20 >= a.dwell_sec:
        dw += 1
    dn = None if dd is None or np.isnan(dd) else float(dd)
    bdir = float(int(vid) % 10) if vid is not None else dn
    nm = A.DIR_NAMES.get((r_, bdir), '') if bdir is not None else ''
    if S is not None and S.variant != 'main':
        nm += f' ({S.variant})'
    return {'r': r_, 'dir': nm, 'dn': vid if vid is not None else dn, 'variant': S.variant if S is not None else None,
            'ty': ty, 'dk': round(dk, 1) if dk >= A.DETOUR_MIN_KM else 0, 'cov': None if cov is None else int(round(100 * cov)),
            't0': int(x.sod.values[0]), 't1': int(x.sod.values[-1]),
            'w0': int(round((int(x.sod.values[0]) - first) / 60)), 'w1': int(round((last - int(x.sod.values[-1])) / 60)),
            'km': round(hav_m(x.lat.values, x.lon.values) / 1000, 1),
            'avg': round(float(sp.mean()), 1), 'dw': dw, 'p': pts}


def build_vehicles(df, a, days, kinds, out, shapes):
    """Історія кожного ТЗ за останні --vehicle-days днів: data/v/<день>/<id>.json + vehicles.json.
    Кожен відрізок обрізається до реального виїзду/приїзду; короткі й неповні позначаються ty='part'."""
    vd = days[-a.vehicle_days:]
    rshapes = {}
    for (rr, _), lst in shapes.items():
        rshapes.setdefault(rr, []).extend(lst)
    sub = df[df.day.isin(vd)].sort_values(['raw_id', 'ts'])
    index = []
    for rid, g in sub.groupby('raw_id'):
        plate = str(g.plate.dropna().mode().iloc[0]) if g.plate.notna().any() else str(rid)
        r0 = g.route.iloc[0]
        entry = {'id': rid, 'plate': plate, 'kind': kinds.get(r0) or SM.kind_of(g.vtype.iloc[0], r0),
                 'routes': sorted(g.route.unique()), 'days': []}
        for day, gd in g.groupby('day'):
            ts, rt = gd.ts.values, gd.route.values
            dr = smooth_dir(gd.dir.values)
            # розрив: пропуск даних >10 хв, зміна маршруту або напрямку
            brk = np.r_[True, (np.diff(ts) > 600) | (rt[1:] != rt[:-1]) | (dr[1:] != dr[:-1])]
            starts = np.flatnonzero(brk)
            segs = []
            for lo, hi in zip(starts, np.r_[starts[1:], len(ts)]):
                if hi - lo < 3:
                    continue
                x = gd.iloc[lo:hi]
                dd = None if np.isnan(dr[lo]) else float(dr[lo])
                S, vid = None, dd
                cl = rshapes.get(rt[lo])
                if cl:        # варіант траси для цього відрізка — за фактичним шляхом і чинністю в цей день
                    best, bs = None, 0.3
                    for v_, S_ in cl:
                        if not A.shape_valid(S_, day):
                            continue
                        sc = SM._variant_score(S_, x.lat.values, x.lon.values, x.speed.fillna(0).values, x.ts.values)
                        if sc > bs:
                            best, bs = (v_, S_), sc
                    if best:
                        vid, S = best
                if S is None:
                    sg = make_seg(x, None, None, rt[lo], dd, None, a)
                    if sg:
                        segs.append(sg)
                    continue
                s, d = S.project(x.lat.values, x.lon.values)
                on = np.flatnonzero(d <= A.D_MAX_M)
                cuts = on[np.flatnonzero(np.diff(s[on]) < -0.5 * S.L) + 1] if len(on) > 1 else []   # нове коло / новий рейс
                bounds = [0, *[int(c) for c in cuts], len(x)]
                for b0, b1 in zip(bounds[:-1], bounds[1:]):
                    if b1 - b0 < 3:
                        continue
                    sg = make_seg(x.iloc[b0:b1], s[b0:b1], d[b0:b1], rt[lo], dd, S, a, vid)
                    if sg:
                        segs.append(sg)
            if not segs:
                continue
            segs.sort(key=lambda z: z['t0'])
            d_ = os.path.join(out, 'data', 'v', day)
            os.makedirs(d_, exist_ok=True)
            dump(os.path.join(d_, f'{rid}.json'), {'id': rid, 'plate': plate, 'day': day, 'segs': segs})
            entry['days'].append(day)
        if entry['days']:
            index.append(entry)
    index.sort(key=lambda e: (e['kind'], e['plate']))
    dump(os.path.join(out, 'vehicles.json'), index)
    print(f'ТЗ з історією: {len(index)} (дні: {", ".join(vd)})')


# ------------------------------ зупинки (static) ------------------------------
def load_stops(static_path):
    if os.path.isdir(static_path):
        files = sorted(__import__('glob').glob(os.path.join(static_path, 'static-*.zip')))
        if not files:
            return None
        static_path = files[-1]
    z = zipfile.ZipFile(static_path)
    names = set(z.namelist())
    if not {'stops.txt', 'stop_times.txt', 'trips.txt', 'routes.txt'} <= names:
        print('  ⚠️ у static немає stops/stop_times — зупинок на карті не буде')
        return None
    rd = lambda n, **k: pd.read_csv(z.open(n), dtype=str, encoding='utf-8-sig', **k)
    stops, routes, trips = rd('stops.txt'), rd('routes.txt'), rd('trips.txt')
    st = rd('stop_times.txt', usecols=['trip_id', 'stop_id', 'stop_sequence'])
    routes['short'] = routes.route_short_name.str.strip()
    t = trips.merge(routes[['route_id', 'short']], on='route_id')
    dirc = pd.to_numeric(t['direction_id'], errors='coerce') if 'direction_id' in t else np.nan
    t['dir'] = (dirc if isinstance(dirc, pd.Series) else pd.Series(np.nan, index=t.index))
    t['dir'] = t['dir'].fillna(t.trip_id.str.extract(r'_([01])$')[0].astype(float))
    # які маршрути зупиняються на зупинці
    j = st[['trip_id', 'stop_id']].merge(t[['trip_id', 'short']], on='trip_id')[['stop_id', 'short']].drop_duplicates()
    by_stop = j.groupby('stop_id').short.apply(list).to_dict()
    # еталонний рейс для кожного (маршрут, напрям): найдовший за кількістю зупинок
    cnt = st.groupby('trip_id').size().rename('n').reset_index()
    t = t.merge(cnt, on='trip_id').dropna(subset=['dir']).sort_values('n', ascending=False)
    rep = t.drop_duplicates(['short', 'dir'])
    seqs = {}
    for r in rep.itertuples():
        g = st[st.trip_id == r.trip_id].copy()
        g['q'] = g.stop_sequence.astype(float)
        seqs[(r.short, float(r.dir))] = g.sort_values('q').stop_id.tolist()
    sd = stops.set_index('stop_id')
    info = {sid: (row.stop_name, float(row.stop_lat), float(row.stop_lon)) for sid, row in sd.iterrows()}
    return info, by_stop, seqs


def match_short(route, shorts):
    p = A.parse_route(route)
    for sh in shorts:
        if A.route_match(A.parse_route(sh), p):
            return sh
    return None


def analytics_data(df, a, days):
    """analytics_data.json: кількість ТЗ на лінії по днях (5-хв кроки) та виконання плану випуску — для вкладки «Аналітика»."""
    pv, _, _ = A.vehicles_tables(df[df.day.isin(days)])
    uniq = df[df.day.isin(days)].groupby(['route', 'day']).raw_id.nunique()
    routes = {}
    for (r, day, dt), row in pv.iterrows():
        routes.setdefault(r, {})[day] = {'dt': dt, 'v': [int(x) for x in row.values], 'u': int(uniq.get((r, day), 0))}
    plan = {}
    if a.plan:
        try:
            pb = A.plan_by_day(a.plan, pv, uniq.to_dict())
            for rec in pb.itertuples():
                for n_ in rec.names:
                    plan.setdefault(n_, []).append({'day': rec.day, 'dt': rec.day_type, 'planned': rec.planned,
                                                    'peak': rec.peak, 'unique': rec.unique})
        except Exception as e:
            print(f'  план не прочитано: {e}')
    dt_ua = {d: A.DAY_TYPE_UA.get(d, d) for d in set(pv.index.get_level_values('day_type'))}
    dump(os.path.join(a.out, 'analytics_data.json'), {'routes': routes, 'plan': plan, 'dt': dt_ua, 'days': days})
    print(f'  Аналітика по днях: {len(routes)} маршрутів' + (f', план для {len(plan)}' if plan else ''))


def publish_analytics(out, rep='report'):
    """Копіює графіки analyze1 (report/*.png) у site/analytics/ і пише analytics.json для вкладки «Аналітика»."""
    if not os.path.isdir(rep):
        print('  report/ не знайдено — вкладка «Аналітика» буде порожня')
        return
    dst = os.path.join(out, 'analytics')
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    routes, general = {}, []
    for f in sorted(os.listdir(rep)):
        if not f.endswith('.png'):
            continue
        shutil.copy(os.path.join(rep, f), os.path.join(dst, f))
        stem = f[:-4]
        for pre, key in (('trip_time_', 'trip_time'),):
            if stem.startswith(pre):
                routes.setdefault(stem[len(pre):], {})[key] = f
                break
        else:
            title = {'speed_by_hour': 'Швидкість по годинах за видами ТЗ', 'plan_vs_fact': 'Плановий випуск і факт'}.get(
                stem, 'Якість даних ' + stem.replace('quality_', '') if stem.startswith('quality_') else stem)
            general.append({'title': title, 'file': f})
    dump(os.path.join(out, 'analytics.json'), {'routes': routes, 'general': general})
    print(f'  Аналітика: {len(routes)} маршрутів, {len(general)} загальних графіків → {dst}')


def main():
    ap = argparse.ArgumentParser(description='Підготовка даних для вебдашборду')
    ap.add_argument('--data', default='.')
    ap.add_argument('--static', default='')
    ap.add_argument('--lines', '--routes-dir', dest='routes_dir', default='')
    ap.add_argument('--out', default='site')
    ap.add_argument('--days', default='')
    ap.add_argument('--day-types', default='')
    ap.add_argument('--routes', default='all')
    ap.add_argument('--bin-m', type=int, default=100)
    ap.add_argument('--min-n', type=int, default=8)
    ap.add_argument('--dmax', type=float, default=40)
    ap.add_argument('--replay-days', type=int, default=2, help='скільки останніх днів покласти в Replay')
    ap.add_argument('--vehicle-days', type=int, default=3, help='скільки останніх днів зберігати в історії ТЗ')
    ap.add_argument('--dwell-sec', type=int, default=120, help='стоянка довша за це (с) = простій')
    ap.add_argument('--plan', default='', help='CSV з плановим випуском (route,type,day_type,planned,valid_from,valid_to) — для вкладки «Аналітика»')
    ap.add_argument('--start-zone', type=float, default=150, help='зона виїзду біля початку траси, м')
    ap.add_argument('--end-zone', type=float, default=120, help='зона приїзду біля кінця траси, м')
    ap.add_argument('--detour', default='', help="період об'їзду РРРР-ММ-ДД:РРРР-ММ-ДД; дані за нього — окремий тип дня «(об’їзд)»")
    ap.add_argument('--detour-routes', default='', help="маршрути, яких стосується об'їзд, через кому (порожньо = усі); номери без літери — автобуси")
    ap.add_argument('--dwell-min-count', type=int, default=2)
    ap.add_argument('--min-cov', type=float, default=0.85, help='частка траси, яку має охоплювати повний рейс (решта = короткі/неповні відрізки)')
    a = ap.parse_args()

    A.START_ZONE_M, A.END_ZONE_M = a.start_zone, a.end_zone
    if os.path.exists(os.path.join('report', 'unmatched_trips.csv')):
        os.remove(os.path.join('report', 'unmatched_trips.csv'))
    only_days = {d.strip() for d in a.days.split(',') if d.strip()}
    print('Завантаження:')
    df = A.load_positions(a.data, only_days)
    ov = A.load_day_overrides(a.day_types)
    df['day_type'] = A.assign_day_types(df, ov, a.detour, a.detour_routes)
    loc = pd.to_datetime(df.ts, unit='s', utc=True).dt.tz_convert('Europe/Kyiv')
    df['sod'] = loc.dt.hour * 3600 + loc.dt.minute * 60 + loc.dt.second
    df['hour'] = loc.dt.hour
    df_all = df.copy()                       # вся доба: для рейсів, ТЗ по датах і плану
    df = df[(df.hour >= 6) & (df.hour < 22)].copy()
    days = sorted(df.day.unique())
    dtypes = sorted(df.day_type.unique())
    print(f'Дні: {", ".join(days)}')

    if a.static:
        A.load_dir_names(a.static, df)
    own = A.load_routes_dir(a.routes_dir) if a.routes_dir else {}
    static = A.load_static(a.static) if a.static else {}
    if not own and not static:
        sys.exit('Потрібні траси: --static і/або --lines')
    st = load_stops(a.static) if a.static else None

    all_routes = sorted(df.route.unique())
    routes = all_routes if a.routes.lower() in ('all', 'всі', '') else A.resolve_routes(a.routes.split(','), all_routes)
    vt = df.groupby('route').vtype.agg(lambda s_: s_.mode().iloc[0])
    kinds = {r: SM.kind_of(vt.get(r, ''), r) for r in routes}

    # час рейсу двома методами
    print('Час рейсу за траєкторією (за лініями):')
    shape_sets = [('lines', own), ('static', static)]
    geo = A.geometry_trips(df_all, [x for x in shape_sets if x[1]], routes)
    geo = A.clean_trips(geo, 'за трасою') if not geo.empty else geo
    med = lambda t_, r, dr: (None if t_.empty else t_[(t_.route == r) & (t_.dir == dr)])

    os.makedirs(os.path.join(a.out, 'data'), exist_ok=True)
    index = {'routes': [], 'days': days, 'daytypes': [{'id': d, 'name': DT_UA.get(d, d)} for d in dtypes],
             'windows': list(WINS), 'bin_m': a.bin_m, 'generated': datetime.now().strftime('%Y-%m-%d %H:%M')}
    stop_routes = {}          # stop_id -> set(імена маршрутів з даних)
    n_files = 0
    shape_map = {}            # (маршрут, напрям) -> Shape, для обрізання рейсів ТЗ
    for r in routes:
        shapes = (A.get_shapes(own, r) if own else None) or (A.get_shapes(static, r) if static else None)
        if not shapes:
            print(f'  {r}: траси немає — пропускаю')
            continue
        sub = df[df.route == r]
        dirs_out = []
        g_all = sub.sort_values(['raw_id', 'ts']).reset_index(drop=True)
        lab, cands, unm = SM.pick_variants(shapes, g_all)      # напрям і варіант — за геометрією, а не за міткою dir
        if (lab < 0).any():
            why = {1: 'надто мало точок на лінії (<8)', 2: 'поза лінією (гараж, об’їзд, інший шлях)', 3: 'немає чинного варіанта лінії в цей день',
                   4: 'по лінії, але рух не вздовж її напрямку (мішанина напрямів, петля)'}
            tot = {}
            for u in unm:
                tot[u[2]] = tot.get(u[2], 0) + (u[1] - u[0])
            parts = ', '.join(f'{100 * v / len(g_all):.0f}% {why[c]}' for c, v in sorted(tot.items(), key=lambda kv: -kv[1]) if v >= 0.005 * len(g_all))
            print(f'  {r}: поза жодною лінією {100 * float((lab < 0).mean()):.0f}% точок — {parts}; список у report/unmatched_trips.csv')
            os.makedirs('report', exist_ok=True)
            pth = os.path.join('report', 'unmatched_trips.csv')
            new_ = not os.path.exists(pth)
            hm = lambda t: pd.to_datetime(t, unit='s', utc=True).tz_convert('Europe/Kyiv').strftime('%H:%M')
            with open(pth, 'a', encoding='utf-8-sig') as fh:
                if new_:
                    fh.write('route,raw_id,day,from,to,points,reason,share_on_line,cover,forward_share,points_on_line\n')
                for a_, b_, c_, on_, cv_, fw_, no_ in unm:
                    fh.write(f'{r},{g_all.raw_id.iloc[a_]},{g_all.day.iloc[a_]},{hm(int(g_all.ts.iloc[a_]))},{hm(int(g_all.ts.iloc[b_ - 1]))},'
                             f'{b_ - a_},{why[c_]},{on_:.2f},{cv_:.2f},{fw_:.2f},{no_}\n')
        dirs_all = g_all.dir.values.astype(float)
        cdir = {}
        for ci in sorted(set(lab.tolist())):
            if ci < 0 or (lab == ci).sum() < 50:
                continue
            dv = dirs_all[lab == ci]
            cdir[ci] = (1 if np.nanmean(dv) > 0.5 else 0) if np.isfinite(dv).any() else int(shapes[ci][1])
        by_sd = {}
        for ci, d_ in cdir.items():
            by_sd.setdefault(d_, set()).add(int(shapes[ci][1]))
        if any(len(v) > 1 for v in by_sd.values()):      # мітки dir у даних плутані — беремо напрям із файлу ліній (forward=0, backward=1)
            cdir = {ci: int(shapes[ci][1]) for ci in cdir}
        by_dr = {}
        for ci, d_ in cdir.items():
            by_dr.setdefault(d_, []).append((int((lab == ci).sum()), ci))
        if not by_dr:
            print(f'  {r}: трасу не підібрано')
            continue
        for dr, lst in sorted(by_dr.items()):
            used = [ci for _, ci in sorted(lst, reverse=True)]
            g0 = g_all
            for k, vi in enumerate(used):
                g, S = g0[lab == vi].reset_index(drop=True), cands[vi]
                vid = int(dr) + 10 * k               # унікальний id напрямку+варіанта (k=0 — основний, id = напрям)
                vname = '' if (len(used) == 1 and S.variant == 'main') else f' ({S.variant})'
                shape_map.setdefault((r, int(dr)), []).append((vid, S))
                s, d = S.project(g.lat.values, g.lon.values)
                v = g.speed.values.astype(float)
                hour, sod = g.hour.values, g.sod.values
                dtype, day = g.day_type.values, g.day.values
                keep = (d <= a.dmax) & (s >= A.ZONE_M) & (s <= S.L - A.ZONE_M) & np.isfinite(v)
                rp = {k: v_ for k, v_ in replay_tracks(g, s, d, sod, v, day).items() if k in days[-a.replay_days:]}
                dd = {'dir': vid, 'base_dir': int(dr), 'variant': S.variant, 'name': A.DIR_NAMES.get((r, float(dr)), f'напрям {int(dr)}') + vname,
                      'km': round(S.L / 1000, 2), 'L': int(S.L), 'line': line_of(S),
                      'speed': speed_bins(S, s, d, v, hour, dtype, dtypes, a),
                      'timeline': timeline(S, s, d, v, sod, dtype, dtypes, a),
                      'replay': rp, 'dwell': dwell_points(S, g, s, d, a),
                      'stats': {'mean': round(float(v[keep].mean()), 1) if keep.any() else None,
                                'stopped': round(100 * float((v[keep] < 3).mean())) if keep.any() else None,
                                'points': int(keep.sum())}}
                tt = {}
                if not geo.empty:
                    x = geo[(geo.route == r) & (geo.dir == dr) & (geo.variant == S.variant)]
                    xd = geo[(geo.route == r) & (geo.dir == dr) & (geo.variant == A.detour_variant(S.variant))]
                    if len(x):
                        tt['geometry'] = {'median': round(float(x.total_min.median()), 1), 'moving': round(float(x.moving_min.median()), 1), 'n': int(len(x))}
                    if len(xd):
                        tt['detour'] = {'median': round(float(xd.total_min.median()), 1), 'moving': round(float(xd.moving_min.median()), 1), 'n': int(len(xd))}
                dd['trip_time'] = tt
                if st:
                    info, by_stop, seqs = st
                    sh = match_short(r, {k[0] for k in seqs})
                    if sh and (sh, float(dr)) in seqs:
                        chain = []
                        for sid in seqs[(sh, float(dr))]:
                            if sid not in info:
                                continue
                            nm, la, lo = info[sid]
                            sp, dp = S.project(np.array([la]), np.array([lo]))
                            chain.append([sid, nm, round(la, 5), round(lo, 5), int(sp[0]) if dp[0] < 120 else None])
                        dd['stops'] = chain
                dirs_out.append(dd)
                print(f'  {r} → {dd["name"]}: {dd["km"]} км, середня {dd["stats"]["mean"]} км/год')
        if not dirs_out:
            continue
        fid = f'r{len(index["routes"]):03d}'
        dump(os.path.join(a.out, 'data', fid + '.json'), {'route': r, 'kind': kinds[r], 'dirs': dirs_out})
        index['routes'].append({'id': r, 'kind': kinds[r], 'file': fid,
                                'dirs': [{'dir': x['dir'], 'name': x['name'], 'km': x['km']} for x in dirs_out]})
        n_files += 1

    if st:
        info, by_stop, seqs = st
        shorts = {k[0] for k in seqs} | {s_ for v_ in by_stop.values() for s_ in v_}
        data_names = {sh: next((x['id'] for x in index['routes'] if A.route_match(A.parse_route(sh), A.parse_route(x['id']))), sh)
                      for sh in shorts}
        stops_out = []
        for sid, (nm, la, lo) in info.items():
            rs = sorted({data_names.get(s_, s_) for s_ in by_stop.get(sid, [])})
            if rs:
                stops_out.append([sid, nm, round(la, 5), round(lo, 5), rs])
        dump(os.path.join(a.out, 'stops.json'), stops_out)
        print(f'Зупинок: {len(stops_out)}')
    dump(os.path.join(a.out, 'index.json'), index)
    build_vehicles(df_all, a, days, kinds, a.out, shape_map)
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, 'index.html')):
        shutil.copy(os.path.join(here, 'index.html'), os.path.join(a.out, 'index.html'))

    print(f'\nГотово: маршрутів {n_files}. Перегляд: python -m http.server -d {a.out}  →  http://localhost:8000')

    # --- Автоматичний запуск analyze1.py та speed_map.py ---
    common_args = ['--data', a.data]
    if a.days:
        common_args += ['--days', a.days]
    if a.day_types:
        common_args += ['--day-types', a.day_types]
    if a.detour:
        common_args += ['--detour', a.detour, '--detour-routes', a.detour_routes]
    if a.static:
        common_args += ['--static', a.static]
    if a.routes_dir:
        common_args += ['--routes-dir', a.routes_dir]

    print('\n[1/2] Запуск analyze1.py (звіти та графіки)...')
    analyze_cmd = [sys.executable, os.path.join(here, 'analyze1.py'), *common_args, '--route', a.routes, '--out', 'report',
                   '--start-zone', str(a.start_zone), '--end-zone', str(a.end_zone)]
    if a.plan:
        analyze_cmd += ['--plan', a.plan]
    subprocess.run(analyze_cmd, check=False)

    publish_analytics(a.out)
    analytics_data(df_all, a, days)

    if a.static or a.routes_dir:
        print('\n[2/2] Запуск speed_map.py (карта швидкостей)...')
        sm_cmd = [sys.executable, os.path.join(here, 'speed_map.py'), *common_args, '--routes', a.routes, '--out', 'speedmap']
        subprocess.run(sm_cmd, check=False)


if __name__ == '__main__':
    main()

## Можете замінити вміст файлу на цей код і запускати `python build_site_2.py --data . --static static.zip --lines lines --out site`
## python -m http.server -d site 8000 