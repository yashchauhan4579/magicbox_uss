import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { API_BASE_URL } from '../../config';

const POLL_MS = 5000;
const useLiveClock = () => {
  const [now, setNow] = React.useState(new Date());
  React.useEffect(() => { const id = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(id); }, []);
  return now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true, timeZone: 'Asia/Kolkata' }).toUpperCase();
};
const cn = (...c) => c.filter(Boolean).join(' ');
const fmtHour = (h) => {
  const ampm = h >= 12 ? 'PM' : 'AM';
  const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h;
  return `${h12}${ampm}`;
};
const fmtTime = (iso) => {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', timeZone: 'Asia/Kolkata' });
  } catch { return ''; }
};
const todayIST = () => new Date().toLocaleDateString('en-CA', { timeZone: 'Asia/Kolkata' });
const ago = (iso) => {
  if (!iso) return '';
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
};

export default function MagicBoxCrowdDashboard() {
  const liveClock = useLiveClock();
  const [devices, setDevices] = useState([]);
  const [fleet, setFleet] = useState([]);
  const [frames, setFrames] = useState([]);
  const [frameLimit, setFrameLimit] = useState(24);
  const [search, setSearch] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [selectedStation, setSelectedStation] = useState(null);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [selectedHour, setSelectedHour] = useState(null);
  const [hourlyData, setHourlyData] = useState(null);
  const [historyFrames, setHistoryFrames] = useState([]);
  const [zoomFrame, setZoomFrame] = useState(null);
  const [zoomIdx, setZoomIdx] = useState(-1);
  const [viewMode, setViewMode] = useState('all'); // 'all' | 'detected'
  const [detectedFrames, setDetectedFrames] = useState([]);
  const [detectedVisible, setDetectedVisible] = useState(24);
  const [locFilter, setLocFilter] = useState([]);
  const [camFilter, setCamFilter] = useState([]);
  const [showLocDrop, setShowLocDrop] = useState(false);
  const [expandedLocs, setExpandedLocs] = useState([]);
  const [locSearch, setLocSearch] = useState("");
  const locDropRef = useRef(null);
  const searchRef = useRef(null);

  /* fleet metadata (on mount) */
  useEffect(() => {
    fetch(`${API_BASE_URL}/magicbox-crowd/fleet`)
      .then(r => r.ok ? r.json() : { stations: [] })
      .then(d => setFleet(d.stations || []))
      .catch(() => {});
  }, []);

  /* poll status + frames */
  useEffect(() => {
    let alive = true;
    const poll = () => {
      fetch(`${API_BASE_URL}/magicbox-crowd/status`)
        .then(r => r.ok ? r.json() : { devices: [] })
        .then(d => alive && setDevices(d.devices || []))
        .catch(() => {});
      if (selectedHour === null) {
        fetch(`${API_BASE_URL}/magicbox-crowd/frames?limit=${frameLimit}`)
          .then(r => r.ok ? r.json() : { frames: [] })
          .then(d => alive && setFrames(d.frames || []))
          .catch(() => {});
      }
    };
    poll();
    const iv = setInterval(poll, POLL_MS);
    return () => { alive = false; clearInterval(iv); };
  }, [frameLimit, selectedHour]);

  /* close location dropdown on outside click */
  useEffect(() => {
    const handler = (e) => { if (locDropRef.current && !locDropRef.current.contains(e.target)) setShowLocDrop(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* clear location search when dropdown closes */
  useEffect(() => { if (!showLocDrop) setLocSearch(""); }, [showLocDrop]);

  /* reset detected pagination on mode/filter change */
  useEffect(() => { setDetectedVisible(24); }, [viewMode, selectedHour]);

  /* clear location filters when leaving detected mode */
  useEffect(() => {
    if (viewMode !== 'detected') { setLocFilter([]); setCamFilter([]); setShowLocDrop(false); }
  }, [viewMode]);

  /* compute device IPs for location filter */
  const locDeviceIds = useMemo(() => {
    if (viewMode !== 'detected' || (locFilter.length === 0 && camFilter.length === 0)) return '';
    const ips = new Set();
    for (const st of fleet) {
      if (locFilter.includes(st.name)) {
        for (const d of st.devices || []) ips.add(d.ip);
      }
    }
    for (const key of camFilter) {
      ips.add(key.split('::')[0]);
    }
    return [...ips].join(',');
  }, [viewMode, locFilter, camFilter, fleet]);

  /* fetch hourly data */
  useEffect(() => {
    const today = todayIST();
    const params = new URLSearchParams({ date: today });
    if (selectedDevice) params.set('device_id', selectedDevice);
    if (selectedCamera) params.set('camera', selectedCamera);
    if (locDeviceIds) params.set('device_ids', locDeviceIds);
    const doFetch = () => fetch(`${API_BASE_URL}/magicbox-crowd/hourly?${params}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setHourlyData(d))
      .catch(() => {});
    doFetch();
    const iv = setInterval(doFetch, 30000);
    return () => clearInterval(iv);
  }, [selectedDevice, selectedCamera, locDeviceIds]);

  /* fetch history frames when hour selected */
  useEffect(() => {
    if (selectedHour === null) { setHistoryFrames([]); return; }
    const today = todayIST();
    const params = new URLSearchParams({ date: today, hour: selectedHour, limit: '30' });
    if (selectedDevice) params.set('device_id', selectedDevice);
    else if (selectedStation) {
      const st = fleet.find(s => s.name === selectedStation);
      if (st) params.set('device_ids', (st.devices || []).map(d => d.ip).join(','));
    }
    if (selectedCamera) params.set('camera', selectedCamera);
    fetch(`${API_BASE_URL}/magicbox-crowd/frames-history?${params}`)
      .then(r => r.ok ? r.json() : { frames: [] })
      .then(d => setHistoryFrames(d.frames || []))
      .catch(() => {});
  }, [selectedHour, selectedDevice, selectedCamera, selectedStation, fleet]);

  /* fetch detected frames (from disk, heads > 0) */
  useEffect(() => {
    if (viewMode !== 'detected') return;
    const today = todayIST();
    const params = new URLSearchParams({ date: today, min_heads: '1', limit: String(detectedVisible) });
    if (selectedDevice) params.set('device_id', selectedDevice);
    else if (selectedStation) {
      const st = fleet.find(s => s.name === selectedStation);
      if (st) params.set('device_ids', (st.devices || []).map(d => d.ip).join(','));
    }
    if (selectedCamera) params.set('camera', selectedCamera);
    if (selectedHour !== null) params.set('hour', selectedHour);
    if (locDeviceIds) params.set('device_ids', locDeviceIds);
    fetch(`${API_BASE_URL}/magicbox-crowd/frames-history?${params}`)
      .then(r => r.ok ? r.json() : { frames: [] })
      .then(d => setDetectedFrames(d.frames || []))
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode, selectedDevice, selectedCamera, selectedHour, detectedVisible, selectedStation, locDeviceIds]);

  /* close search dropdown on outside click */
  useEffect(() => {
    const handler = (e) => { if (searchRef.current && !searchRef.current.contains(e.target)) setSearchOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* keyboard for zoom modal */
  useEffect(() => {
    if (zoomFrame === null) return;
    const handler = (e) => {
      if (e.key === 'Escape') { setZoomFrame(null); setZoomIdx(-1); }
      if (e.key === 'ArrowRight') navZoom(1);
      if (e.key === 'ArrowLeft') navZoom(-1);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [zoomFrame, zoomIdx]);

  /* device→metadata lookup from fleet */
  const meta = useMemo(() => {
    const m = {};
    for (const st of fleet) {
      for (const dev of st.devices || []) {
        m[dev.ip] = { name: dev.name, station: st.name, cameras: (dev.cameras || []).map(c => c.name) };
      }
    }
    return m;
  }, [fleet]);

  const merged = useMemo(() => devices.map(d => ({ ...d, meta: meta[d.device_id] || null })), [devices, meta]);

  /* search results */
  const searchResults = useMemo(() => {
    if (!search || search.length < 2) return [];
    const q = search.toLowerCase();
    const results = [];
    for (const st of fleet) {
      const stMatch = st.name.toLowerCase().includes(q);
      const matchingDevs = (st.devices || []).filter(d => d.name.toLowerCase().includes(q) || d.ip.includes(q));
      if (stMatch || matchingDevs.length > 0) {
        results.push({ station: st.name, devices: stMatch ? (st.devices || []) : matchingDevs });
      }
      if (results.length >= 12) break;
    }
    return results;
  }, [search, fleet]);

  /* station list with live counts */
  const stationData = useMemo(() => {
    const map = {};
    for (const st of fleet) {
      map[st.name] = { name: st.name, totalDevices: (st.devices || []).length, totalCameras: (st.devices || []).reduce((s, d) => s + (d.cameras || []).length, 0), online: 0, heads: 0 };
    }
    for (const d of merged) {
      const st = d.meta?.station || 'Unknown';
      if (!map[st]) map[st] = { name: st, totalDevices: 0, totalCameras: 0, online: 0, heads: 0 };
      if (d.is_online) map[st].online++;
      map[st].heads += (d.total_heads || 0);
    }
    return Object.values(map).sort((a, b) => b.heads - a.heads || b.online - a.online || a.name.localeCompare(b.name));
  }, [merged, fleet]);

  const stationDevices = useMemo(() => {
    if (!selectedStation) return [];
    const fleetSt = fleet.find(s => s.name === selectedStation);
    const allDevs = []; const seen = new Set();
    for (const d of merged) {
      if ((d.meta?.station || 'Unknown') === selectedStation) { allDevs.push(d); seen.add(d.device_id); }
    }
    if (fleetSt) {
      for (const fd of fleetSt.devices || []) {
        if (!seen.has(fd.ip)) {
          allDevs.push({ device_id: fd.ip, is_online: false, total_heads: 0, cameras: fd.cameras.map(c => ({ name: c.name, grab_ok: false, head_count: 0 })), meta: { name: fd.name, station: selectedStation } });
        }
      }
    }
    return allDevs;
  }, [merged, fleet, selectedStation]);

  const deviceCameras = useMemo(() => {
    if (!selectedDevice) return [];
    const dev = merged.find(d => d.device_id === selectedDevice);
    if (dev?.cameras?.length) return dev.cameras;
    for (const st of fleet) { for (const fd of st.devices || []) { if (fd.ip === selectedDevice) return fd.cameras.map(c => ({ name: c.name, grab_ok: false, head_count: 0 })); } }
    return [];
  }, [merged, fleet, selectedDevice]);

  /* global stats */
  const totalActiveCameras = useMemo(() => merged.reduce((s, d) => s + (d.cameras?.filter(c => c.grab_ok)?.length || 0), 0), [merged]);
  const totalCameras = useMemo(() => fleet.length > 0 ? fleet.reduce((s, st) => s + (st.devices || []).reduce((s2, d) => s2 + (d.cameras || []).length, 0), 0) : merged.reduce((s, d) => s + (d.cameras?.length || 0), 0), [merged, fleet]);
  const totalHeads = useMemo(() => merged.reduce((s, d) => s + (d.total_heads || 0), 0), [merged]);
  const totalOnline = merged.filter(d => d.is_online).length;
  const totalDevices = useMemo(() => fleet.length > 0 ? fleet.reduce((s, st) => s + (st.devices || []).length, 0) : merged.length, [merged, fleet]);

  /* filtered stats */
  const filteredDevices = useMemo(() => !selectedStation ? merged : merged.filter(d => (d.meta?.station || 'Unknown') === selectedStation), [merged, selectedStation]);
  const filteredCamCount = useMemo(() => {
    if (selectedCamera) return 1;
    if (selectedDevice) return filteredDevices.find(d => d.device_id === selectedDevice)?.cameras?.length || 0;
    return filteredDevices.reduce((s, d) => s + (d.cameras?.length || 0), 0);
  }, [filteredDevices, selectedDevice, selectedCamera]);
  const filteredHeads = useMemo(() => {
    if (selectedCamera && selectedDevice) { const cam = filteredDevices.find(d => d.device_id === selectedDevice)?.cameras?.find(c => c.name === selectedCamera); return cam?.head_count || 0; }
    if (selectedDevice) return filteredDevices.find(d => d.device_id === selectedDevice)?.total_heads || 0;
    return filteredDevices.reduce((s, d) => s + (d.total_heads || 0), 0);
  }, [filteredDevices, selectedDevice, selectedCamera]);

  /* station IPs for filtering (stable — only changes when fleet/selectedStation change) */
  const stationIps = useMemo(() => {
    if (!selectedStation) return null;
    const ips = new Set();
    for (const d of merged) {
      if ((d.meta?.station || 'Unknown') === selectedStation) ips.add(d.device_id);
    }
    // Also check fleet for offline devices
    for (const st of fleet) {
      if (st.name === selectedStation) {
        for (const d of st.devices || []) ips.add(d.ip);
      }
    }
    return ips;
  }, [selectedStation, fleet]);

  /* active frames (live, history, or detected) */
  const activeFrames = useMemo(() => {
    let source;
    if (viewMode === 'detected') {
      source = detectedFrames;
    } else if (selectedHour !== null) {
      source = historyFrames;
    } else {
      source = frames;
    }
    let list = source;
    if (selectedStation && stationIps) {
      list = list.filter(f => stationIps.has(f.device_id));
    }
    if (selectedDevice && viewMode !== 'detected') list = list.filter(f => f.device_id === selectedDevice);
    if (selectedCamera && viewMode !== 'detected') list = list.filter(f => f.camera_name === selectedCamera);
    // Location-wise filter (detected mode only)
    if (viewMode === 'detected' && (locFilter.length > 0 || camFilter.length > 0)) {
      const camFilterMatch = new Set();
      for (const k of camFilter) {
        const parts = k.split('::');
        camFilterMatch.add(parts[0] + '::' + parts[1]);
      }
      list = list.filter(f => {
        const station = meta[f.device_id]?.station || 'Unknown';
        const camKey = f.device_id + '::' + (f.camera_name || '');
        if (camFilterMatch.size > 0 && camFilterMatch.has(camKey)) return true;
        if (camFilterMatch.size > 0 && !locFilter.includes(station)) return false;
        if (locFilter.length > 0) return locFilter.includes(station);
        return true;
      });
    }
    return list;
  }, [frames, historyFrames, detectedFrames, stationIps, selectedStation, selectedDevice, selectedCamera, selectedHour, viewMode, locFilter, camFilter, meta]);

  const selectionLabel = useMemo(() => {
    if (selectedCamera && selectedDevice) return `${selectedCamera} — ${meta[selectedDevice]?.name || selectedDevice}`;
    if (selectedDevice) return meta[selectedDevice]?.name || selectedDevice;
    if (selectedStation) return selectedStation;
    return null;
  }, [selectedStation, selectedDevice, selectedCamera, meta]);

  /* nav helpers */
  const selectStation = useCallback((st) => { setSelectedStation(st); setSelectedDevice(null); setSelectedCamera(null); setSelectedHour(null); setSearch(''); setSearchOpen(false); }, []);
  const selectDevice = useCallback((did) => { setSelectedDevice(did); setSelectedCamera(null); setSelectedHour(null); setSearch(''); setSearchOpen(false); }, []);
  const clearAll = useCallback(() => { setSelectedStation(null); setSelectedDevice(null); setSelectedCamera(null); setSelectedHour(null); }, []);
  const loadMore = useCallback(() => setFrameLimit(l => Math.min(l + 24, 100)), []);

  /* zoom nav */
  const openZoom = useCallback((f, idx) => { setZoomFrame(f); setZoomIdx(idx); }, []);
  const navZoom = useCallback((dir) => {
    setZoomIdx(prev => {
      const next = prev + dir;
      if (next < 0 || next >= activeFrames.length) return prev;
      setZoomFrame(activeFrames[next]);
      return next;
    });
  }, [activeFrames]);

  /* hourly chart data */
  const maxHourHeads = useMemo(() => {
    if (!hourlyData?.hours) return 1;
    return Math.max(1, ...hourlyData.hours.map(h => h.total_heads));
  }, [hourlyData]);
  const maxHourFrames = useMemo(() => {
    if (!hourlyData?.hours) return 1;
    return Math.max(1, ...hourlyData.hours.map(h => h.frame_count));
  }, [hourlyData]);

  return (
    <div className="flex h-full text-white font-sans text-sm overflow-hidden">
      {/* ── SIDEBAR ── */}
      <div className="w-72 shrink-0 border-r border-white/10 flex flex-col bg-black/40 backdrop-blur-md">
        {/* search */}
        <div className="p-4 border-b border-white/10 relative" ref={searchRef}>
          <input
            className="w-full bg-white/5 border border-white/15 rounded-lg px-3 py-2 text-sm placeholder:text-white/30 focus:outline-none focus:border-violet-400/60 focus:ring-1 focus:ring-violet-400/30"
            placeholder="Search location or device..."
            value={search}
            onChange={e => { setSearch(e.target.value); setSearchOpen(true); }}
            onFocus={() => search.length >= 2 && setSearchOpen(true)}
          />
          {searchOpen && searchResults.length > 0 && (
            <div className="absolute left-4 right-4 top-full mt-1 bg-gray-900/95 border border-white/15 rounded-xl shadow-2xl max-h-72 overflow-y-auto z-50 backdrop-blur-lg">
              {searchResults.map((r, i) => (
                <div key={i}>
                  <button onClick={() => selectStation(r.station)} className="w-full text-left px-4 py-2.5 hover:bg-violet-500/20 border-b border-white/5">
                    <span className="text-violet-300 text-sm font-semibold">{r.station}</span>
                    <span className="text-white/40 text-xs ml-2">{r.devices.length} device{r.devices.length !== 1 ? 's' : ''}</span>
                  </button>
                  {r.devices.slice(0, 4).map((d, j) => (
                    <button key={j} onClick={() => { selectStation(r.station); setTimeout(() => selectDevice(d.ip), 50); }}
                      className="w-full text-left px-6 py-1.5 hover:bg-white/5 text-xs">
                      <span className="text-white/80">{d.name}</span>
                      <span className="text-white/30 ml-2">{d.ip}</span>
                      <span className="text-white/30 ml-1">· {(d.cameras || []).length} cam{(d.cameras || []).length !== 1 ? 's' : ''}</span>
                    </button>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* breadcrumb */}
        <div className="px-4 py-2.5 border-b border-white/10 flex items-center gap-1.5 text-xs flex-wrap">
          <button onClick={clearAll} className={cn('hover:text-violet-300 transition-colors', !selectedStation && 'text-violet-300 font-bold')}>All ({totalDevices})</button>
          {selectedStation && (<><span className="text-white/20">/</span><button onClick={() => { setSelectedDevice(null); setSelectedCamera(null); }} className={cn('hover:text-violet-300 truncate max-w-[140px]', !selectedDevice && 'text-violet-300 font-bold')}>{selectedStation}</button></>)}
          {selectedDevice && (<><span className="text-white/20">/</span><button onClick={() => setSelectedCamera(null)} className={cn('hover:text-violet-300 truncate max-w-[120px]', !selectedCamera && 'text-violet-300 font-bold')}>{(meta[selectedDevice]?.name || selectedDevice).split(',')[0]}</button></>)}
          {selectedCamera && (<><span className="text-white/20">/</span><span className="text-violet-300 font-bold truncate max-w-[100px]">{selectedCamera}</span></>)}
        </div>

        {/* list */}
        <div className="flex-1 overflow-y-auto">
          {!selectedStation ? (
            stationData.map(st => (
              <button key={st.name} onClick={() => selectStation(st.name)} className="w-full text-left px-4 py-3 hover:bg-white/5 border-b border-white/[0.06] group transition-colors">
                <div className="flex items-center justify-between">
                  <span className="text-sm truncate group-hover:text-violet-300 max-w-[160px] transition-colors">{st.name}</span>
                  {st.heads > 0 && <span className="text-violet-400 font-bold">{st.heads}</span>}
                </div>
                <div className="flex gap-3 text-xs text-white/40 mt-1">
                  <span>{st.online > 0 ? <span className="text-emerald-400 font-medium">{st.online}</span> : '0'}/{st.totalDevices} online</span>
                  <span>{st.totalCameras} cams</span>
                </div>
              </button>
            ))
          ) : !selectedDevice ? (
            stationDevices.map(d => (
              <button key={d.device_id} onClick={() => selectDevice(d.device_id)} className="w-full text-left px-4 py-3 hover:bg-white/5 border-b border-white/[0.06] group transition-colors">
                <div className="flex items-center gap-2">
                  <span className={cn('w-2 h-2 rounded-full shrink-0', d.is_online ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-red-400/40')} />
                  <span className="text-sm truncate group-hover:text-violet-300">{d.meta?.name || d.device_id}</span>
                  {d.total_heads > 0 && <span className="ml-auto text-violet-400 font-bold">{d.total_heads}</span>}
                </div>
                <div className="flex gap-3 text-xs text-white/35 mt-1 pl-4">
                  <span>{d.device_id}</span>
                  <span>{d.cameras?.length || 0} cams</span>
                </div>
              </button>
            ))
          ) : (
            deviceCameras.map((cam, i) => (
              <button key={cam.name || i} onClick={() => setSelectedCamera(selectedCamera === cam.name ? null : cam.name)}
                className={cn('w-full text-left px-4 py-3 hover:bg-white/5 border-b border-white/[0.06] group transition-colors', selectedCamera === cam.name && 'bg-violet-500/10 border-l-2 border-l-violet-400')}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={cn('w-2 h-2 rounded-full', cam.grab_ok ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-red-400/40')} />
                    <span className={cn('text-sm', selectedCamera === cam.name && 'text-violet-300 font-semibold')}>{cam.name}</span>
                  </div>
                  <span className="text-violet-400 font-bold text-base">{cam.grab_ok ? cam.head_count : '—'}</span>
                </div>
                <div className="text-xs text-white/30 pl-4 mt-0.5">{cam.grab_ok ? `${cam.inference_ms}ms inference` : 'OFFLINE'}</div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* ── MAIN ── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* stats bar */}
        <div className="flex-1 overflow-y-auto">
        <div className="px-5 py-3 border-b border-white/10 bg-gradient-to-r from-black/30 via-violet-950/10 to-black/30">
          <div className="flex items-center gap-8 flex-wrap">
            <div className="flex flex-col">
              <span className="text-violet-400 font-bold text-lg tracking-wider">MAGICBOX CROWD</span>
              <span className="text-white/50 text-xs font-mono tracking-widest">{liveClock}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse shadow-md shadow-emerald-400/40" />
              <span className="text-emerald-300 font-semibold text-base">{totalOnline}</span>
              <span className="text-white/40">/ {totalDevices} Online</span>
            </div>
            <div className="h-5 w-px bg-white/15" />
            <div>
              <span className="text-white font-bold text-base">{totalActiveCameras}</span>
              <span className="text-white/40"> / {totalCameras} Cameras</span>
            </div>
            <div className="h-5 w-px bg-white/15" />
            <div className="bg-violet-500/15 rounded-lg px-3 py-1">
              <span className="text-violet-300 font-bold text-xl">{totalHeads}</span>
              <span className="text-violet-300/60 text-sm ml-1">Live Now</span>
            </div>
            <div className="h-5 w-px bg-white/15" />
            <div className="bg-emerald-500/10 rounded-lg px-3 py-1">
              <span className="text-emerald-300 font-bold text-xl">{hourlyData?.hours ? hourlyData.hours.reduce((s, h) => s + h.total_heads, 0) : '–'}</span>
              <span className="text-emerald-300/60 text-sm ml-1">Today's Detections</span>
            </div>
          </div>

          {(selectionLabel || selectedHour !== null) && (
            <div className="flex items-center gap-4 mt-2 text-sm">
              {selectionLabel && (<><span className="text-white/40">Viewing:</span><span className="text-violet-300 font-semibold">{selectionLabel}</span><span className="text-white/15">|</span><span className="text-white/80 font-medium">{filteredCamCount}</span><span className="text-white/40"> cam{filteredCamCount !== 1 ? 's' : ''}</span><span className="text-white/15">|</span><span className="text-violet-400 font-bold text-lg">{filteredHeads}</span><span className="text-white/40"> heads</span></>)}
              {selectedHour !== null && (<><span className="text-white/15">|</span><span className="bg-violet-500/20 text-violet-300 rounded px-2 py-0.5 text-xs font-medium">{fmtHour(selectedHour)} – {fmtHour(selectedHour + 1)}</span></>)}
              <button onClick={clearAll} className="ml-auto text-white/40 hover:text-white/70 text-xs border border-white/15 rounded-lg px-3 py-1 hover:bg-white/5 transition-colors">Clear all</button>
            </div>
          )}
        </div>

        {/* hourly chart */}
        {hourlyData?.hours && (
          <div className="px-5 py-3 border-b border-white/10 bg-black/20">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-white/50 font-medium uppercase tracking-wide">Hourly Activity — {hourlyData.date}</span>
              {selectedHour !== null && (
                <button onClick={() => setSelectedHour(null)} className="text-xs text-violet-300/70 hover:text-violet-300">Show all hours</button>
              )}
            </div>
            <div className="flex items-end gap-[3px] h-20">
              {hourlyData.hours.map(h => {
                const pct = Math.max(2, (h.frame_count / maxHourFrames) * 100);
                const isActive = selectedHour === h.hour;
                const hasCounts = h.total_heads > 0;
                return (
                  <button key={h.hour} onClick={() => setSelectedHour(selectedHour === h.hour ? null : h.hour)}
                    className={cn('flex-1 rounded-t transition-all relative group cursor-pointer min-w-0',
                      isActive ? 'bg-violet-400 shadow-lg shadow-violet-500/30' : hasCounts ? 'bg-violet-500/60 hover:bg-violet-400/70' : 'bg-white/8 hover:bg-white/15'
                    )} style={{ height: `${pct}%` }} title={`${fmtHour(h.hour)}: ${h.total_heads} heads, ${h.frame_count} frames`}>
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-gray-900/95 text-[10px] text-white/80 px-2 py-1 rounded shadow-lg whitespace-nowrap hidden group-hover:block z-10 pointer-events-none border border-white/10">
                      {fmtHour(h.hour)} · {h.total_heads} heads · {h.frame_count} frames
                    </div>
                  </button>
                );
              })}
            </div>
            <div className="flex gap-[3px] mt-1">
              {hourlyData.hours.map(h => (
                <div key={h.hour} className="flex-1 text-center text-[9px] text-white/25 min-w-0">
                  {h.hour % 3 === 0 ? fmtHour(h.hour) : ''}
                </div>
              ))}
              <div className="text-[9px] text-white/25 w-[18px] text-center shrink-0">12AM</div>
            </div>
          </div>
        )}

        {/* view mode toggle + detection summary */}
        <div className="px-5 py-2 border-b border-white/10 bg-black/10">
          <div className="flex items-center gap-2 flex-wrap">
            <div className="flex bg-white/5 rounded-lg p-0.5 border border-white/10">
              <button onClick={() => setViewMode('all')}
                className={cn('px-4 py-1.5 rounded-md text-xs font-medium transition-all', viewMode === 'all' ? 'bg-violet-500/30 text-violet-300 shadow-sm' : 'text-white/40 hover:text-white/60')}>
                All Frames
              </button>
              <button onClick={() => setViewMode('detected')}
                className={cn('px-4 py-1.5 rounded-md text-xs font-medium transition-all flex items-center gap-1.5', viewMode === 'detected' ? 'bg-emerald-500/30 text-emerald-300 shadow-sm' : 'text-white/40 hover:text-white/60')}>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Detected
              </button>
            </div>

            {/* selected location/camera chips */}
            {viewMode === 'detected' && (locFilter.length > 0 || camFilter.length > 0) && (() => {
              const chips = [];
              for (const stName of locFilter) chips.push({ type: 'loc', station: stName, label: 'All cameras' });
              if (camFilter.length > 0) {
                const byStation = {};
                for (const key of camFilter) {
                  const parts = key.split('::');
                  const station = meta[parts[0]]?.station || 'Unknown';
                  if (!byStation[station]) byStation[station] = [];
                  byStation[station].push({ cam: parts[1], key });
                }
                for (const [station, cams] of Object.entries(byStation))
                  for (const c of cams) chips.push({ type: 'cam', station, label: c.cam, camKey: c.key });
              }
              return chips.map((chip, i) => (
                <button key={i} onClick={() => {
                  if (chip.type === 'loc') { setLocFilter(locFilter.filter(x => x !== chip.station)); }
                  else { setCamFilter(camFilter.filter(x => x !== chip.camKey)); }
                }} className="group flex items-center gap-1 bg-violet-500/10 border border-violet-400/20 rounded-lg px-2 py-1 hover:bg-violet-500/20 hover:border-violet-400/40 transition-all cursor-pointer">
                  <span className="text-[11px] text-violet-300 font-medium">{chip.station}</span>
                  <span className="text-[10px] text-white/40">{chip.label}</span>
                  <svg className="w-2.5 h-2.5 text-white/30 group-hover:text-red-400 transition-colors shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                </button>
              ));
            })()}

            {/* Location-wise filter dropdown — only in detected mode */}
            {viewMode === 'detected' && (
              <div className="relative ml-auto" ref={locDropRef}>
                <button onClick={() => setShowLocDrop(v => !v)}
                  className={cn('flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-all',
                    (locFilter.length > 0 || camFilter.length > 0)
                      ? 'bg-violet-500/20 border-violet-400/40 text-violet-300'
                      : 'bg-white/5 border-white/10 text-white/50 hover:text-white/70 hover:border-white/20')}>
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a2 2 0 01-2.828 0l-4.243-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                  Location-wise
                  {(locFilter.length > 0 || camFilter.length > 0) && (
                    <span className="bg-violet-400/30 text-violet-200 rounded-full px-1.5 text-[10px] font-bold">{locFilter.length + camFilter.length}</span>
                  )}
                </button>

                {showLocDrop && (
                  <div className="absolute right-0 top-full mt-1 w-72 max-h-80 overflow-y-auto bg-gray-900/98 border border-white/15 rounded-xl shadow-2xl shadow-black/60 z-50 backdrop-blur-xl">
                    <div className="flex items-center justify-between px-3 py-2 border-b border-white/10">
                      <span className="text-xs text-white/60 font-medium uppercase tracking-wider">Filter by Location</span>
                      {(locFilter.length > 0 || camFilter.length > 0) && (
                        <button onClick={() => { setLocFilter([]); setCamFilter([]); }}
                          className="text-[10px] text-violet-400 hover:text-violet-300">Clear all</button>
                      )}
                    </div>
                    <div className="px-2 py-1.5 border-b border-white/10">
                      <input value={locSearch} onChange={e => setLocSearch(e.target.value)} placeholder="Search..." autoFocus
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white/80 placeholder:text-white/25 focus:outline-none focus:border-violet-400/50" />
                    </div>
                    {fleet.filter(st => {
                      if (!locSearch.trim()) return true;
                      const q = locSearch.toLowerCase();
                      if (st.name.toLowerCase().includes(q)) return true;
                      return (st.devices || []).some(d => (d.cameras || []).some(c => c.name.toLowerCase().includes(q)));
                    }).map(st => {
                      const stSelected = locFilter.includes(st.name);
                      const expanded = expandedLocs.includes(st.name);
                      const stCams = (st.devices || []).flatMap(d => (d.cameras || []).map((c, ci) => ({ key: d.ip + '::' + c.name + '::' + ci, label: c.name, device: d.name })));
                      const activeCamCount = stCams.filter(c => camFilter.includes(c.key)).length;
                      return (
                        <div key={st.name} className="border-b border-white/5 last:border-0">
                          <div className="flex items-center gap-2 px-3 py-2 hover:bg-white/5 cursor-pointer">
                            <button onClick={() => {
                              setLocFilter(stSelected ? locFilter.filter(x => x !== st.name) : [...locFilter, st.name]);
                              setCamFilter(camFilter.filter(x => !stCams.some(c => c.key === x)));
                            }} className={cn('w-4 h-4 rounded border flex items-center justify-center shrink-0 transition-all',
                              stSelected ? 'bg-violet-500 border-violet-400' : activeCamCount > 0 ? 'bg-violet-500/40 border-violet-400/50' : 'border-white/20 hover:border-white/40')}>
                              {stSelected && <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                              {!stSelected && activeCamCount > 0 && <svg className="w-2.5 h-2.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" /></svg>}
                            </button>
                            <span className={cn('text-xs flex-1 truncate', stSelected || activeCamCount > 0 ? 'text-white/90' : 'text-white/60')}>{st.name}</span>
                            {activeCamCount > 0 && <span className="text-[10px] text-violet-400">{activeCamCount} cam{activeCamCount > 1 ? 's' : ''}</span>}
                            <button onClick={() => {
                              setExpandedLocs(expanded ? expandedLocs.filter(x => x !== st.name) : [...expandedLocs, st.name]);
                            }} className="text-white/30 hover:text-white/60 transition-colors p-0.5">
                              <svg className={cn('w-3 h-3 transition-transform', expanded && 'rotate-180')} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                            </button>
                          </div>
                          {expanded && (
                            <div className="pb-1">
                              {stCams.map(cam => {
                                const camSel = camFilter.includes(cam.key);
                                return (
                                  <button key={cam.key} onClick={() => {
                                    setCamFilter(camSel ? camFilter.filter(x => x !== cam.key) : [...camFilter, cam.key]);
                                    setLocFilter(locFilter.filter(x => x !== st.name));
                                  }} className={cn('flex items-center gap-2 w-full px-3 pl-9 py-1.5 text-left hover:bg-white/5 transition-colors')}>
                                    <span className={cn('w-3.5 h-3.5 rounded border flex items-center justify-center shrink-0 transition-all',
                                      camSel ? 'bg-emerald-500 border-emerald-400' : 'border-white/15 hover:border-white/30')}>
                                      {camSel && <svg className="w-2 h-2 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" /></svg>}
                                    </span>
                                    <span className={cn('text-[11px] truncate', camSel ? 'text-white/90' : 'text-white/50')}>{cam.label}</span>
                                    <span className="text-[9px] text-white/25 ml-auto truncate max-w-[80px]">{cam.device}</span>
                                  </button>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
          </div>



        </div>

        {/* frames grid */}
        <div className="p-4">
          {activeFrames.length === 0 ? (
            <div className="text-white/30 text-center mt-24 text-base">
              {viewMode === 'detected' ? 'No detections found yet for this filter' : selectedHour !== null ? `No frames for ${fmtHour(selectedHour)} – ${fmtHour(selectedHour + 1)}` : frames.length === 0 ? 'Waiting for devices to report...' : 'No frames for this selection'}
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
                {activeFrames.map((f, i) => (
                  <FrameCard key={`${f.device_id}-${f.camera_name}-${f.timestamp}-${i}`} f={f} meta={meta} onClick={() => openZoom(f, i)} />
                ))}
              </div>
              {viewMode === 'detected' && detectedFrames.length >= detectedVisible && (
                <div className="text-center mt-5">
                  <button onClick={() => setDetectedVisible(v => v + 24)} className="text-violet-300 hover:text-violet-200 text-sm border border-violet-400/30 rounded-lg px-5 py-2 hover:bg-violet-500/10 transition-colors">
                    Load more frames
                  </button>
                </div>
              )}
              {viewMode !== 'detected' && selectedHour === null && frameLimit < 100 && activeFrames.length >= frameLimit - 5 && (
                <div className="text-center mt-5">
                  <button onClick={loadMore} className="text-violet-300 hover:text-violet-200 text-sm border border-violet-400/30 rounded-lg px-5 py-2 hover:bg-violet-500/10 transition-colors">Load more frames</button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* ── ZOOM MODAL ── */}
      </div>

        {zoomFrame && (
        <div className="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm flex items-center justify-center" onClick={() => { setZoomFrame(null); setZoomIdx(-1); }}>
          <div className="relative max-w-[90vw] max-h-[90vh] flex flex-col items-center" onClick={e => e.stopPropagation()}>
            {zoomFrame.frame_b64 ? (
              <img src={`data:image/jpeg;base64,${zoomFrame.frame_b64}`} alt={zoomFrame.camera_name} className="max-w-full max-h-[75vh] rounded-lg shadow-2xl object-contain" />
            ) : (
              <div className="w-96 h-64 flex items-center justify-center bg-gray-900 rounded-lg border border-white/10 text-white/30">Frame expired (JPEG purged after 3hrs)</div>
            )}
            <div className="mt-3 bg-black/60 rounded-xl px-5 py-3 text-center border border-white/10">
              <div className="flex items-center gap-4 justify-center">
                <span className="text-violet-300 font-semibold text-base">{zoomFrame.camera_name}</span>
                <span className="text-white/40">·</span>
                <span className="text-white/70">{meta[zoomFrame.device_id]?.name || zoomFrame.device_id}</span>
                <span className="text-white/40">·</span>
                <span className="text-white/50">{meta[zoomFrame.device_id]?.station || ''}</span>
              </div>
              <div className="flex items-center gap-4 justify-center mt-1">
                <span className={cn('font-bold text-lg', zoomFrame.head_count > 0 ? 'text-violet-400' : 'text-white/40')}>{zoomFrame.head_count} head{zoomFrame.head_count !== 1 ? 's' : ''}</span>
                <span className="text-white/40">·</span>
                <span className="text-white/50">{fmtTime(zoomFrame.timestamp)}</span>
                <span className="text-white/30 text-xs">{ago(zoomFrame.timestamp)}</span>
              </div>
            </div>
            {/* nav arrows */}
            {zoomIdx > 0 && (
              <button onClick={() => navZoom(-1)} className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-14 w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/70 hover:text-white text-xl transition-colors">&larr;</button>
            )}
            {zoomIdx < activeFrames.length - 1 && (
              <button onClick={() => navZoom(1)} className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-14 w-10 h-10 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/70 hover:text-white text-xl transition-colors">&rarr;</button>
            )}
            <button onClick={() => { setZoomFrame(null); setZoomIdx(-1); }} className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center text-white/60 hover:text-white text-sm transition-colors">&times;</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Frame card ───────────────────────────────────────────────────── */
function FrameCard({ f, meta, onClick }) {
  const devMeta = meta[f.device_id];
  const devName = (devMeta?.name || f.device_id).split(',')[0];
  const station = devMeta?.station || '';

  return (
    <div onClick={onClick} className="rounded-xl border border-white/10 bg-white/[0.04] overflow-hidden hover:border-violet-400/40 hover:shadow-lg hover:shadow-violet-500/5 transition-all cursor-pointer group">
      <div className="relative aspect-square bg-black">
        {f.frame_b64 ? (
          <img src={`data:image/jpeg;base64,${f.frame_b64}`} alt={f.camera_name} className="w-full h-full object-cover group-hover:brightness-110 transition-all" loading="lazy" />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-900/80 text-white/20 text-xs">Frame expired</div>
        )}
        <div className={cn(
          'absolute top-2 right-2 px-2.5 py-1 rounded-lg text-xs font-bold shadow-lg',
          f.head_count > 0 ? 'bg-violet-500 text-white shadow-violet-500/40' : 'bg-black/70 text-white/50'
        )}>{f.head_count}</div>
        <div className="absolute bottom-1.5 left-1.5 px-2 py-0.5 rounded-md bg-black/70 text-[10px] text-white/60">{fmtTime(f.timestamp) || ago(f.timestamp)}</div>
      </div>
      <div className="px-3 py-2">
        <div className="text-sm text-white/90 truncate font-semibold">{f.camera_name}</div>
        <div className="text-xs text-white/40 truncate mt-0.5">{devName}</div>
        {station && <div className="text-xs text-violet-400/50 truncate">{station}</div>}
      </div>
    </div>
  );
}
