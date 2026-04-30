import React, { useState, useEffect, useRef } from 'react';
import { API_BASE_URL } from '../../config';

/* ── colour helpers ── */
const COND_COLORS = {
  Clear:    { bg: 'bg-green-500', text: 'text-green-400', hex: '#22c55e' },
  Low:      { bg: 'bg-lime-500',  text: 'text-lime-400',  hex: '#84cc16' },
  Medium:   { bg: 'bg-yellow-500',text: 'text-yellow-400',hex: '#eab308' },
  High:     { bg: 'bg-orange-500',text: 'text-orange-400',hex: '#ea580c' },
  Critical: { bg: 'bg-red-500',   text: 'text-red-400',   hex: '#dc2626' },
};
const cc = (cond) => COND_COLORS[cond] || COND_COLORS.Clear;

/* ── sparkline ── */
const Sparkline = ({ data = [], color = '#0ea5e9', h = 48 }) => {
  if (data.length < 2) return null;
  const max = Math.max(...data, 1);
  const min = Math.min(...data, 0);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    return `${x},${y}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 100 ${h}`} preserveAspectRatio="none" style={{ width: '100%', height: h }} className="block">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
      <polyline points={`0,${h} ${pts} 100,${h}`} fill={color} fillOpacity="0.10" stroke="none" />
    </svg>
  );
};

/* ════════════════════════════════════════════════════════
   Full camera view:
     TOP-LEFT  = CRT video    TOP-RIGHT = analysis sidebar
     BOTTOM    = stats strip
   ════════════════════════════════════════════════════════ */
const _lastHeatmapUrls = {};
const FullCameraView = ({ cameraId, data }) => {
  const c = cc(data.condition);
  const rawUrl = data.heatmap_url
    ? `${API_BASE_URL}/heatmaps/${data.heatmap_url.split('/heatmaps/').pop()}`
    : '';
  if (rawUrl) _lastHeatmapUrls[cameraId] = rawUrl;
  const heatmapUrl = rawUrl || _lastHeatmapUrls[cameraId] || '';
  const congestionPct = ((data.congestion_score || 0) / 10) * 100;

  return (
    <div className="w-full h-full flex flex-col">
      {/* ── TOP ROW: video + analysis sidebar ── */}
      <div className="flex flex-1 min-h-0">

        {/* ── VIDEO with CRT effect ── */}
        <div className="flex-1 min-w-0 flex items-center justify-center bg-[#030508] p-4">
          <div className="relative w-full h-full max-w-[960px]" style={{ perspective: '900px' }}>
            {/* CRT frame */}
            <div
              className="relative w-full h-full overflow-hidden"
              style={{
                borderRadius: '18px / 14px',
                boxShadow: `
                  0 0 40px rgba(20,184,166,0.15),
                  0 0 80px rgba(20,184,166,0.05),
                  inset 0 0 60px rgba(0,0,0,0.5),
                  inset 0 0 120px rgba(0,0,0,0.3)
                `,
                border: '3px solid rgba(255,255,255,0.06)',
              }}
            >
              {/* the feed — heatmap snapshot from crowd-worker */}
              <img
                src={heatmapUrl}
                key={heatmapUrl}
                alt={data.camera_name}
                className="w-full h-full object-cover"
              />

              {/* CRT scanlines */}
              <div
                className="absolute inset-0 pointer-events-none z-10"
                style={{
                  background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)',
                }}
              />
              {/* CRT vignette */}
              <div
                className="absolute inset-0 pointer-events-none z-10"
                style={{
                  background: 'radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%)',
                }}
              />
              {/* CRT glare */}
              <div
                className="absolute inset-0 pointer-events-none z-10"
                style={{
                  background: 'linear-gradient(135deg, rgba(255,255,255,0.03) 0%, transparent 40%, transparent 60%, rgba(255,255,255,0.01) 100%)',
                }}
              />

              {/* live badge */}
              <div className="absolute top-3 left-3 z-20 bg-black/70 backdrop-blur-sm border border-teal-500/30 px-2.5 py-1 flex items-center gap-2 rounded">
                <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                <span className="text-[9px] text-teal-400 font-bold uppercase tracking-wider">{data.camera_name}</span>
              </div>
            </div>
          </div>
        </div>

        {/* ── RIGHT: analysis sidebar ── */}
        <div className="w-[320px] flex-shrink-0 bg-[#060a12] border-l border-white/[0.06] overflow-y-auto">
          <div className="p-4 space-y-4">

            {/* hero: condition + count */}
            <div className="rounded-xl overflow-hidden">
              <div className={`${c.bg} px-5 py-4 text-center`}>
                <div className="text-3xl font-black text-white uppercase tracking-wider">{data.condition}</div>
              </div>
              <div className="bg-white/[0.03] border border-white/[0.06] border-t-0 rounded-b-xl px-5 py-5 text-center">
                <div className="text-5xl font-black text-teal-400 leading-none">{data.count}</div>
                <div className="text-[10px] text-white/30 font-bold uppercase tracking-widest mt-2">Persons Detected</div>
              </div>
            </div>

            {/* congestion bar */}
            <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[9px] text-white/30 font-bold uppercase tracking-widest">Congestion</span>
                <span className="text-2xl font-black text-white">{data.congestion_score}<span className="text-sm text-white/30">/10</span></span>
              </div>
              <div className="h-2.5 bg-white/[0.06] rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    data.congestion_score <= 3 ? 'bg-green-500' :
                    data.congestion_score <= 5 ? 'bg-yellow-500' :
                    data.congestion_score <= 7 ? 'bg-orange-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${congestionPct}%` }}
                />
              </div>
            </div>

            {/* AI behavior */}
            <div className="bg-violet-500/[0.05] border border-violet-500/15 rounded-xl p-4">
              <div className="text-[9px] text-violet-400 font-bold uppercase tracking-widest mb-2">AI Analysis</div>
              <div className="text-[11px] text-white/70 leading-relaxed">{data.behavior}</div>
            </div>

            {/* prediction */}
            <div className="bg-emerald-500/[0.05] border border-emerald-500/15 rounded-xl p-4">
              <div className="text-[9px] text-emerald-400 font-bold uppercase tracking-widest mb-2">Next Prediction</div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-black text-white">~{data.predicted_count}</span>
                <span className="text-xs text-white/30">persons</span>
                <span className={`text-sm font-black ml-auto ${cc(data.predicted_condition).text}`}>{data.predicted_condition}</span>
              </div>
            </div>

            {/* persons trend */}
            {data.counts_history && data.counts_history.length > 1 && (
              <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                <div className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-2">Persons Trend</div>
                <Sparkline data={data.counts_history} color={c.hex} h={56} />
              </div>
            )}

            {/* overall risk + movement + precaution */}
            <div className={`border rounded-xl p-4 space-y-3 ${
              data.overall_risk === 'CRITICAL' ? 'bg-red-500/[0.08] border-red-500/20'
              : data.overall_risk === 'HIGH' ? 'bg-orange-500/[0.08] border-orange-500/20'
              : data.overall_risk === 'MEDIUM' ? 'bg-yellow-500/[0.08] border-yellow-500/20'
              : 'bg-green-500/[0.08] border-green-500/20'
            }`}>
              <div className="flex items-center justify-between">
                <span className="text-[9px] text-white/40 font-bold uppercase tracking-widest">Overall Risk</span>
                <span className={`text-sm font-black ${
                  data.overall_risk === 'CRITICAL' ? 'text-red-400'
                  : data.overall_risk === 'HIGH' ? 'text-orange-400'
                  : data.overall_risk === 'MEDIUM' ? 'text-yellow-400'
                  : 'text-green-400'
                }`}>{data.overall_risk}</span>
              </div>
              <div className="text-[10px] text-white/50 leading-relaxed">{data.sentiment}</div>
              <div className="border-t border-white/[0.06] pt-2">
                <div className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-1">Movement</div>
                <div className="text-[11px] text-white/70 leading-relaxed">{data.flow}</div>
              </div>
              <div className="border-t border-white/[0.06] pt-2">
                <div className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-1">Safety Precaution</div>
                <div className="text-[11px] text-white/70 leading-relaxed">{data.safety_precaution}</div>
              </div>
            </div>

            {/* safety checks */}
            <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4 space-y-2">
              <div className="text-[9px] text-white/30 font-bold uppercase tracking-widest mb-2">Safety Checks</div>
              {[
                { label: 'Weapon', value: data.weapon_detected },
                { label: 'Fight / Injury', value: data.fight_collision_injury },
                { label: 'Wrongful Activity', value: data.wrongful_activity },
              ].map(({ label, value }) => {
                const isYes = String(value || '').toUpperCase().startsWith('YES');
                return (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-[10px] text-white/50">{label}</span>
                    <span className={`text-[10px] font-bold ${isYes ? 'text-red-400' : 'text-green-400'}`}>
                      {isYes ? value : 'NO'}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

/* ════════════════════════════════════════════════════════
   Reports Dropdown
   ════════════════════════════════════════════════════════ */
const _authQuery = () => {
  const token = sessionStorage.getItem('iris_auth_token') || localStorage.getItem('iris_auth_token') || '';
  const tabId = sessionStorage.getItem('iris_tab_id') || '';
  return `auth_token=${encodeURIComponent(token)}&x_client_tab=${encodeURIComponent(tabId)}`;
};

const authFetch = (url, opts = {}) => {
  const sep = url.includes('?') ? '&' : '?';
  return fetch(`${url}${sep}${_authQuery()}`, opts);
};

/* ── Transform crowd-worker data shape to dashboard format ── */
const transformWorkerData = (raw) => {
  const result = {};
  for (const [deviceId, data] of Object.entries(raw)) {
    const level = (data.densityLevel || 'LOW');
    const condMap = { LOW: 'Low', MEDIUM: 'Medium', HIGH: 'High', CRITICAL: 'Critical' };
    const risk = (data.overall_risk || 'LOW').toUpperCase();
    const isCritical = risk === 'CRITICAL' || risk === 'HIGH' || level === 'CRITICAL' || level === 'HIGH';
    result[deviceId] = {
      camera_name: (data.deviceId || deviceId).replace(/^camera_/, '').replaceAll('_', '.'),
      count: data.peopleCount || 0,
      condition: condMap[level] || 'Low',
      congestion_score: data.congestionLevel || 0,
      heatmap_url: data.heatmapImageUrl || '',
      free_space: Math.round(data.freeSpace || 0),
      visibility: data.visibility_score || 90,
      avg_count: data.peopleCount || 0,
      peak_count: data.peopleCount || 0,
      predicted_count: data.predicted_count || data.peopleCount || 0,
      predicted_condition: condMap[level] || 'Low',
      flow: data.crowd_movement || 'Stable movement.',
      behavior: data.behavior || 'Normal activity.',
      // Original pipeline Gemini fields
      crowd_density: data.crowd_density || level,
      sentiment: data.sentiment || 'NEUTRAL',
      weapon_detected: data.weapon_detected || 'NO',
      fight_collision_injury: data.fight_collision_injury || 'NO',
      wrongful_activity: data.wrongful_activity || 'NO',
      safety_precaution: data.safety_precaution || 'Continue standard monitoring.',
      overall_risk: data.overall_risk || 'LOW',
      safety_alert: isCritical ? `${risk} risk — ${data.peopleCount || 0} heads detected` : null,
      safety_level: risk === 'CRITICAL' ? 'Critical' : risk === 'HIGH' ? 'High'
        : risk === 'MEDIUM' ? 'Medium' : 'Low',
      counts_history: [],
      timestamp: data.timestamp || null,
    };
  }
  return result;
};

/* ════════════════════════════════════════════════════════
   Camera Manager — list, add, remove cameras in one place
   ════════════════════════════════════════════════════════ */
const CameraManager = ({ onChanged, fullscreen = false }) => {
  const [tab, setTab] = useState('cameras');
  const [cameras, setCameras] = useState([]);
  const [groups, setGroups] = useState([]);
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [groupName, setGroupName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [removing, setRemoving] = useState(null);
  const [expandedGroup, setExpandedGroup] = useState(null);

  // Upload state
  const [parsedCameras, setParsedCameras] = useState(null);
  const [uploadStep, setUploadStep] = useState('idle'); // idle | parsed | applying
  const uploadRef = useRef(null);

  // Magicboxhub state
  const [mbxStatus, setMbxStatus] = useState(null); // {logged_in, email, has_credentials, base_url}
  const [mbxLoginEmail, setMbxLoginEmail] = useState('');
  const [mbxLoginPassword, setMbxLoginPassword] = useState('');
  const [mbxLoggingIn, setMbxLoggingIn] = useState(false);
  const [mbxTree, setMbxTree] = useState(null); // {stations: [...]}
  const [mbxLoadingTree, setMbxLoadingTree] = useState(false);
  const [mbxTreeError, setMbxTreeError] = useState('');
  const [mbxSelected, setMbxSelected] = useState({}); // { internalName: {rtsp, displayName} }
  const [mbxFilter, setMbxFilter] = useState('');
  const [mbxAdding, setMbxAdding] = useState(false);
  const [mbxAddResult, setMbxAddResult] = useState(null); // {added, skipped, failed}

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true); setError('');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await authFetch(`${API_BASE_URL}/crowd-live/cameras/upload`, {
        method: 'POST', body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setParsedCameras(data.cameras || []);
        setUploadStep('parsed');
      } else {
        const err = await res.json().catch(() => ({}));
        setError(err.detail || 'Failed to parse file');
      }
    } catch (_) { setError('Upload failed'); }
    setBusy(false);
    if (uploadRef.current) uploadRef.current.value = '';
  };

  const handleApplyCameras = async (mode) => {
    if (!parsedCameras) return;
    setUploadStep('applying'); setBusy(true); setError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/cameras/apply`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cameras: parsedCameras, mode }),
      });
      if (res.ok) {
        const data = await res.json();
        setParsedCameras(null);
        setUploadStep('idle');
        await loadCameras();
        if (onChanged) onChanged(data);
        setError('');
      } else {
        const err = await res.json().catch(() => ({}));
        setError(err.detail || 'Failed to apply');
        setUploadStep('parsed');
      }
    } catch (_) { setError('Network error'); setUploadStep('parsed'); }
    setBusy(false);
  };

  const loadCameras = async () => {
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/cameras`);
      if (res.ok) setCameras(await res.json());
    } catch (_) {}
  };
  const loadGroups = async () => {
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/groups`);
      if (res.ok) setGroups(await res.json());
    } catch (_) {}
  };

  useEffect(() => { loadCameras(); loadGroups(); }, []);

  const handleAdd = async () => {
    if (!name.trim() || !url.trim()) return;
    setBusy(true); setError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/cameras/add`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), url: url.trim() }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setName(''); setUrl('');
        if (data.cameras) setCameras(data.cameras);
        if (onChanged) onChanged(data);
      } else setError(data.message || 'Failed');
    } catch (_) { setError('Network error'); }
    setBusy(false);
  };

  const handleRemove = async (camName) => {
    setRemoving(camName);
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/cameras/remove`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: camName }),
      });
      const data = await res.json();
      if (data.status === 'ok' && data.cameras) setCameras(data.cameras);
      if (onChanged) onChanged(data);
    } catch (_) {}
    setRemoving(null);
  };

  const handleCreateGroup = async () => {
    if (!groupName.trim()) return;
    setBusy(true); setError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/groups/create`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: groupName.trim() }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setGroupName('');
        if (data.groups) setGroups(data.groups);
      } else setError(data.message || 'Failed');
    } catch (_) { setError('Network error'); }
    setBusy(false);
  };

  const handleDeleteGroup = async (gName) => {
    setRemoving(gName);
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/groups/delete`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: gName }),
      });
      const data = await res.json();
      if (data.status === 'ok' && data.groups) setGroups(data.groups);
    } catch (_) {}
    setRemoving(null);
  };

  const handleActivateGroup = async (gName) => {
    setBusy(true); setError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/groups/activate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: gName }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        if (data.cameras) setCameras(data.cameras);
        if (onChanged) onChanged(data);
      } else setError(data.message || 'Failed to activate');
    } catch (_) { setError('Network error'); }
    setBusy(false);
  };

  // ── Magicboxhub handlers ──
  const mbxRefreshStatus = async () => {
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/magicbox/status`);
      if (res.ok) {
        const data = await res.json();
        setMbxStatus(data);
        return data;
      }
    } catch (_) {}
    return null;
  };

  const mbxLoadTree = async () => {
    setMbxLoadingTree(true); setMbxTreeError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/magicbox/tree`);
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.status === 'ok') {
        setMbxTree(data);
      } else {
        setMbxTreeError(data.message || `Failed to load camera tree (${res.status})`);
        if (res.status === 401 || (data.message || '').includes('not logged in')) {
          setMbxStatus(s => ({ ...(s || {}), logged_in: false }));
        }
      }
    } catch (_) {
      setMbxTreeError('Network error fetching camera tree');
    }
    setMbxLoadingTree(false);
  };

  const mbxLogin = async () => {
    if (!mbxLoginEmail.trim() || !mbxLoginPassword) return;
    setMbxLoggingIn(true); setMbxTreeError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/magicbox/login`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: mbxLoginEmail.trim(), password: mbxLoginPassword }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.status === 'ok') {
        setMbxStatus(data);
        setMbxLoginPassword('');
        // immediately fetch the tree
        await mbxLoadTree();
      } else {
        setMbxTreeError(data.message || 'Login failed');
      }
    } catch (_) { setMbxTreeError('Network error'); }
    setMbxLoggingIn(false);
  };

  const mbxLogout = async () => {
    try { await authFetch(`${API_BASE_URL}/crowd-live/magicbox/logout`, { method: 'POST' }); } catch (_) {}
    setMbxStatus({ logged_in: false, has_credentials: false, email: null });
    setMbxTree(null);
    setMbxSelected({});
    setMbxAddResult(null);
  };

  const mbxToggleCam = (cam) => {
    if (cam.alreadyAdded) return;
    setMbxSelected(prev => {
      const next = { ...prev };
      if (next[cam.internalName]) delete next[cam.internalName];
      else next[cam.internalName] = { rtsp: cam.rtsp, displayName: cam.name };
      return next;
    });
  };

  const mbxToggleDevice = (device, deviceCams) => {
    const selectableCams = deviceCams.filter(c => !c.alreadyAdded);
    const allSelected = selectableCams.length > 0 && selectableCams.every(c => mbxSelected[c.internalName]);
    setMbxSelected(prev => {
      const next = { ...prev };
      if (allSelected) {
        for (const c of selectableCams) delete next[c.internalName];
      } else {
        for (const c of selectableCams) next[c.internalName] = { rtsp: c.rtsp, displayName: c.name };
      }
      return next;
    });
  };

  const mbxAddSelected = async () => {
    const list = Object.entries(mbxSelected).map(([internalName, v]) => ({
      internalName, rtsp: v.rtsp,
    }));
    if (!list.length) return;
    setMbxAdding(true); setMbxAddResult(null); setMbxTreeError('');
    try {
      const res = await authFetch(`${API_BASE_URL}/crowd-live/magicbox/add`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cameras: list }),
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.status === 'ok') {
        setMbxAddResult({ added: data.added || [], skipped: data.skipped || [], failed: data.failed || [] });
        if (data.cameras) setCameras(data.cameras);
        setMbxSelected({});
        if (onChanged) onChanged(data);
        // Refresh tree so newly-added cams render with `alreadyAdded: true`
        mbxLoadTree();
      } else {
        setMbxTreeError(data.message || 'Bulk add failed');
      }
    } catch (_) { setMbxTreeError('Network error'); }
    setMbxAdding(false);
  };

  // On entering magicbox tab: fetch status (and tree if logged in)
  useEffect(() => {
    if (tab !== 'magicbox') return;
    let alive = true;
    (async () => {
      const s = await mbxRefreshStatus();
      if (!alive) return;
      if (s && s.logged_in) mbxLoadTree();
    })();
    return () => { alive = false; };
  }, [tab]);

  const tabBtn = (id, label) => (
    <button
      onClick={() => { setTab(id); setError(''); }}
      className={`flex-1 py-1.5 text-[9px] font-black uppercase tracking-widest rounded transition-all ${
        tab === id ? 'bg-teal-500/20 text-teal-400 border border-teal-500/30' : 'text-white/30 hover:text-white/50 border border-transparent'
      }`}
    >{label}</button>
  );

  const content = (
    <div className="space-y-3">
      {/* Tab bar */}
      <div className="flex gap-1">
        {tabBtn('cameras', 'Cameras')}
        {tabBtn('magicbox', 'Magicbox')}
        {tabBtn('upload', 'Upload')}
        {tabBtn('groups', 'Groups')}
      </div>

      {tab === 'cameras' && (<>
        {/* Camera list */}
        <div className="flex items-center justify-between">
          <span className="text-[9px] text-white/30">{cameras.length} camera{cameras.length !== 1 ? 's' : ''} active</span>
        </div>
        {cameras.length > 0 && (
          <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
            {cameras.map(cam => (
              <div key={cam.id} className="flex items-center justify-between bg-white/[0.03] border border-white/5 rounded px-3 py-2 group">
                <div className="text-[10px] text-white/80 font-bold truncate flex-1 min-w-0">{cam.name}</div>
                <button
                  onClick={() => handleRemove(cam.name)}
                  disabled={removing === cam.name}
                  className="ml-2 flex-shrink-0 px-2 py-0.5 text-[8px] font-bold uppercase tracking-wider rounded
                    bg-red-500/0 hover:bg-red-500/20 text-white/20 hover:text-red-400 border border-transparent hover:border-red-500/30
                    transition-all opacity-0 group-hover:opacity-100"
                >{removing === cam.name ? '...' : 'Remove'}</button>
              </div>
            ))}
          </div>
        )}
        <div className="border-t border-white/5" />
        <div className="text-[9px] text-white/30 font-bold uppercase tracking-wider">Add Camera</div>
        <input type="text" placeholder="Camera name" value={name} onChange={e => setName(e.target.value)}
          className="w-full bg-white/5 border border-white/10 rounded px-3 py-2 text-[11px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
        <input type="text" placeholder="RTSP URL" value={url} onChange={e => setUrl(e.target.value)}
          className="w-full bg-white/5 border border-white/10 rounded px-3 py-2 text-[11px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
        {error && <div className="text-[9px] text-red-400">{error}</div>}
        <button onClick={handleAdd} disabled={busy || !name.trim() || !url.trim()}
          className={`w-full py-2 rounded font-black text-[10px] uppercase tracking-widest transition-all ${
            !busy && name.trim() && url.trim() ? 'bg-teal-500 hover:bg-teal-400 text-white' : 'bg-white/5 text-white/20 cursor-not-allowed'
          }`}>{busy ? 'Adding...' : 'Add Camera'}</button>
      </>)}

      {tab === 'upload' && (<>
        <div className="text-[9px] text-white/30 font-bold uppercase tracking-wider">Upload Camera List</div>
        <div className="text-[8px] text-white/20">Upload an Excel (.xlsx) or CSV file with camera names and RTSP URLs</div>
        <input ref={uploadRef} type="file" accept=".xlsx,.xls,.csv" onChange={handleFileUpload}
          className="hidden" />
        {uploadStep === 'idle' && (
          <button onClick={() => uploadRef.current?.click()} disabled={busy}
            className="w-full py-3 rounded border-2 border-dashed border-white/10 hover:border-teal-500/40 text-white/30 hover:text-teal-400 text-[10px] font-bold uppercase tracking-wider transition-all">
            {busy ? 'Parsing...' : 'Choose File'}
          </button>
        )}
        {uploadStep === 'parsed' && parsedCameras && (
          <div className="space-y-2">
            <div className="text-[9px] text-teal-400 font-bold">{parsedCameras.length} cameras found</div>
            <div className="max-h-36 overflow-y-auto space-y-1 pr-1">
              {parsedCameras.map((c, i) => (
                <div key={i} className="flex items-center gap-2 bg-white/[0.03] border border-white/5 rounded px-2 py-1.5">
                  <div className="flex-1 min-w-0">
                    <div className="text-[9px] text-white/70 font-bold truncate">{c.name}</div>
                    <div className="text-[7px] text-white/30 truncate">{c.ip}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="border-t border-white/5 pt-2">
              <div className="text-[9px] text-white/40 mb-2">What would you like to do with existing cameras?</div>
              <div className="flex gap-2">
                <button onClick={() => handleApplyCameras('replace')} disabled={busy}
                  className="flex-1 py-2 rounded font-black text-[9px] uppercase tracking-widest bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 border border-orange-500/30 transition-all disabled:opacity-40">
                  {busy ? '...' : 'Replace All'}
                </button>
                <button onClick={() => handleApplyCameras('append')} disabled={busy}
                  className="flex-1 py-2 rounded font-black text-[9px] uppercase tracking-widest bg-teal-500/20 hover:bg-teal-500/30 text-teal-400 border border-teal-500/30 transition-all disabled:opacity-40">
                  {busy ? '...' : 'Add to Existing'}
                </button>
              </div>
              <button onClick={() => { setParsedCameras(null); setUploadStep('idle'); }}
                className="w-full mt-2 py-1.5 text-[8px] text-white/30 hover:text-white/50 transition-colors">
                Cancel
              </button>
            </div>
          </div>
        )}
        {uploadStep === 'applying' && (
          <div className="text-[9px] text-teal-400 text-center py-4">Applying cameras and restarting analytics...</div>
        )}
        {error && <div className="text-[9px] text-red-400">{error}</div>}
      </>)}

      {tab === 'magicbox' && (<>
        {(!mbxStatus || !mbxStatus.logged_in) ? (
          <div className="space-y-2">
            <div className="text-[9px] text-white/30 font-bold uppercase tracking-wider">Magicboxhub Login</div>
            <div className="text-[8px] text-white/30">
              Sign in to {mbxStatus?.base_url || 'app.magicboxhub.net'} to browse the camera tree.
            </div>
            <input type="email" placeholder="Email" value={mbxLoginEmail}
              onChange={e => setMbxLoginEmail(e.target.value)}
              className="w-full bg-white/5 border border-white/10 rounded px-3 py-2 text-[11px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
            <input type="password" placeholder="Password" value={mbxLoginPassword}
              onChange={e => setMbxLoginPassword(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') mbxLogin(); }}
              className="w-full bg-white/5 border border-white/10 rounded px-3 py-2 text-[11px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
            {mbxTreeError && <div className="text-[9px] text-red-400">{mbxTreeError}</div>}
            <button onClick={mbxLogin}
              disabled={mbxLoggingIn || !mbxLoginEmail.trim() || !mbxLoginPassword}
              className={`w-full py-2 rounded font-black text-[10px] uppercase tracking-widest transition-all ${
                !mbxLoggingIn && mbxLoginEmail.trim() && mbxLoginPassword
                  ? 'bg-teal-500 hover:bg-teal-400 text-white'
                  : 'bg-white/5 text-white/20 cursor-not-allowed'
              }`}>
              {mbxLoggingIn ? 'Logging in...' : 'Login'}
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            {/* Header: identity + logout + refresh */}
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-white/40 truncate">
                <span className="text-white/30">as</span> {mbxStatus.email || 'magicboxhub'}
              </span>
              <div className="flex gap-2">
                <button onClick={mbxLoadTree} disabled={mbxLoadingTree}
                  className="text-[8px] text-teal-400 hover:text-teal-300 uppercase tracking-wider font-bold disabled:opacity-40">
                  {mbxLoadingTree ? '...' : 'Refresh'}
                </button>
                <button onClick={mbxLogout}
                  className="text-[8px] text-white/30 hover:text-red-400 uppercase tracking-wider font-bold">
                  Logout
                </button>
              </div>
            </div>

            {/* Filter + add button */}
            <div className="flex gap-2 items-center">
              <input type="text" placeholder="Filter station/device/camera..."
                value={mbxFilter} onChange={e => setMbxFilter(e.target.value)}
                className="flex-1 bg-white/5 border border-white/10 rounded px-3 py-1.5 text-[10px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
              <button onClick={mbxAddSelected}
                disabled={mbxAdding || Object.keys(mbxSelected).length === 0}
                className={`px-3 py-1.5 rounded font-black text-[10px] uppercase tracking-widest transition-all whitespace-nowrap ${
                  !mbxAdding && Object.keys(mbxSelected).length > 0
                    ? 'bg-teal-500 hover:bg-teal-400 text-white'
                    : 'bg-white/5 text-white/20 cursor-not-allowed'
                }`}>
                {mbxAdding ? '...' : `Add (${Object.keys(mbxSelected).length})`}
              </button>
            </div>

            {/* Result toast */}
            {mbxAddResult && (
              <div className="text-[9px] bg-white/[0.03] border border-white/5 rounded px-2 py-1.5 space-y-0.5">
                {mbxAddResult.added.length > 0 && (
                  <div className="text-teal-400">+ {mbxAddResult.added.length} added to crowd analysis</div>
                )}
                {mbxAddResult.skipped.length > 0 && (
                  <div className="text-white/40">{mbxAddResult.skipped.length} already added (skipped)</div>
                )}
                {mbxAddResult.failed.length > 0 && (
                  <div className="text-red-400">{mbxAddResult.failed.length} failed</div>
                )}
              </div>
            )}

            {/* Tree */}
            {mbxTreeError && <div className="text-[9px] text-red-400">{mbxTreeError}</div>}
            {mbxLoadingTree && !mbxTree && (
              <div className="text-[9px] text-white/30 text-center py-6">Loading camera tree...</div>
            )}
            {mbxTree && mbxTree.stations && (() => {
              const filt = mbxFilter.trim().toLowerCase();
              const stationsToRender = mbxTree.stations
                .map(st => {
                  const stHit = !filt || (st.name || '').toLowerCase().includes(filt);
                  const devices = (st.devices || [])
                    .map(dev => {
                      const devHit = stHit || (dev.name || '').toLowerCase().includes(filt)
                        || (dev.ip || '').toLowerCase().includes(filt)
                        || (dev.location || '').toLowerCase().includes(filt);
                      const cams = (dev.cameras || []).filter(c =>
                        devHit || (c.name || '').toLowerCase().includes(filt)
                      );
                      return cams.length ? { ...dev, cameras: cams } : null;
                    })
                    .filter(Boolean);
                  return devices.length ? { ...st, devices } : null;
                })
                .filter(Boolean);

              if (!stationsToRender.length) {
                return <div className="text-[9px] text-white/30 text-center py-6">No cameras match filter.</div>;
              }

              return (
                <div className="max-h-72 overflow-y-auto space-y-2 pr-1">
                  {stationsToRender.map(st => (
                    <div key={st.name} className="bg-white/[0.02] border border-white/5 rounded">
                      <div className="px-2 py-1.5 border-b border-white/5">
                        <div className="text-[10px] text-teal-400 font-bold truncate">{st.name}</div>
                        {st.division && <div className="text-[8px] text-white/30">{st.division}</div>}
                      </div>
                      <div className="p-1.5 space-y-1.5">
                        {st.devices.map(dev => {
                          const selectable = dev.cameras.filter(c => !c.alreadyAdded);
                          const allSelected = selectable.length > 0 && selectable.every(c => mbxSelected[c.internalName]);
                          const anySelected = selectable.some(c => mbxSelected[c.internalName]);
                          return (
                            <div key={dev.id || dev.name} className="bg-white/[0.02] rounded">
                              <div className="flex items-center justify-between px-2 py-1">
                                <div className="min-w-0 flex-1">
                                  <div className="text-[10px] text-white/80 font-bold truncate">{dev.name}</div>
                                  <div className="text-[8px] text-white/30 truncate">
                                    {dev.ip}{dev.location ? ` · ${dev.location}` : ''} · {dev.cameras.length} cam{dev.cameras.length !== 1 ? 's' : ''}
                                  </div>
                                </div>
                                {selectable.length > 0 && (
                                  <button onClick={() => mbxToggleDevice(dev, dev.cameras)}
                                    className="ml-2 text-[8px] text-teal-400 hover:text-teal-300 uppercase tracking-wider font-bold whitespace-nowrap">
                                    {allSelected ? 'Clear' : anySelected ? 'All' : 'All'}
                                  </button>
                                )}
                              </div>
                              <div className="px-2 pb-1.5 space-y-0.5">
                                {dev.cameras.map(cam => {
                                  const checked = !!mbxSelected[cam.internalName] || cam.alreadyAdded;
                                  // Magicbox DB's per-camera status hasn't been refreshed in months
                                  // and reads "offline" for almost every cam. Device status is probed
                                  // every 30s by the hub — far more reliable.
                                  const offline = (dev.status || '').toLowerCase() === 'offline';
                                  return (
                                    <label key={cam.internalName}
                                      className={`flex items-center gap-2 px-2 py-1 rounded text-[10px] ${
                                        cam.alreadyAdded
                                          ? 'bg-white/[0.02] cursor-not-allowed'
                                          : 'bg-white/[0.02] hover:bg-white/5 cursor-pointer'
                                      }`}>
                                      <input type="checkbox" checked={checked} disabled={cam.alreadyAdded}
                                        onChange={() => mbxToggleCam(cam)}
                                        className="accent-teal-500 cursor-pointer disabled:cursor-not-allowed" />
                                      <span className={`flex-1 truncate ${cam.alreadyAdded ? 'text-white/30' : 'text-white/80'}`}>
                                        {cam.name}
                                      </span>
                                      {cam.alreadyAdded && (
                                        <span className="text-[7px] text-teal-400 font-bold uppercase">added</span>
                                      )}
                                      {offline && !cam.alreadyAdded && (
                                        <span className="text-[7px] text-orange-400 font-bold uppercase">offline</span>
                                      )}
                                    </label>
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </div>
        )}
      </>)}

      {tab === 'groups' && (<>
        {/* Groups list */}
        {groups.length > 0 ? (
          <div className="max-h-60 overflow-y-auto space-y-2 pr-1">
            {groups.map(g => (
              <div key={g.name} className="bg-white/[0.03] border border-white/5 rounded overflow-hidden">
                <div className="flex items-center gap-2 px-3 py-2">
                  <button onClick={() => setExpandedGroup(expandedGroup === g.name ? null : g.name)}
                    className="text-[10px] text-white/50 hover:text-white/80 transition-colors">
                    {expandedGroup === g.name ? '▼' : '▶'}
                  </button>
                  <div className="flex-1 min-w-0">
                    <div className="text-[10px] text-white/80 font-bold truncate">{g.name}</div>
                    <div className="text-[8px] text-white/30">{g.count} camera{g.count !== 1 ? 's' : ''}</div>
                  </div>
                  {(() => {
                    const activeNames = new Set(cameras.map(c => c.name));
                    const groupNames = new Set((g.cameras || []).map(c => c.name));
                    const isActive = groupNames.size > 0 && groupNames.size === activeNames.size && [...groupNames].every(n => activeNames.has(n));
                    return isActive ? (
                      <span className="px-2 py-1 text-[8px] font-bold uppercase tracking-wider rounded
                        bg-green-500/20 text-green-400 border border-green-500/30">Active</span>
                    ) : (
                      <button onClick={() => handleActivateGroup(g.name)} disabled={busy}
                        className="px-2 py-1 text-[8px] font-bold uppercase tracking-wider rounded
                          bg-teal-500/20 hover:bg-teal-500/30 text-teal-400 border border-teal-500/30 transition-all">
                        {busy ? '...' : 'Activate'}
                      </button>
                    );
                  })()}
                  <button onClick={() => handleDeleteGroup(g.name)} disabled={removing === g.name}
                    className="px-2 py-1 text-[8px] font-bold uppercase tracking-wider rounded
                      bg-red-500/0 hover:bg-red-500/20 text-white/20 hover:text-red-400 border border-transparent hover:border-red-500/30 transition-all">
                    {removing === g.name ? '...' : '×'}
                  </button>
                </div>
                {expandedGroup === g.name && g.cameras && g.cameras.length > 0 && (
                  <div className="border-t border-white/5 px-3 py-2 space-y-1">
                    {g.cameras.map((c, i) => (
                      <div key={i} className="text-[9px] text-white/40 truncate pl-4">• {c.name}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[9px] text-white/20 text-center py-4">No groups yet</div>
        )}
        <div className="border-t border-white/5" />
        <div className="text-[9px] text-white/30 font-bold uppercase tracking-wider">Save Current as Group</div>
        <div className="flex gap-2">
          <input type="text" placeholder="Group name" value={groupName} onChange={e => setGroupName(e.target.value)}
            className="flex-1 bg-white/5 border border-white/10 rounded px-3 py-2 text-[11px] text-white/90 placeholder-white/20 focus:border-teal-500/50 focus:outline-none" />
          <button onClick={handleCreateGroup} disabled={busy || !groupName.trim()}
            className={`px-4 py-2 rounded font-black text-[10px] uppercase tracking-widest transition-all ${
              !busy && groupName.trim() ? 'bg-teal-500 hover:bg-teal-400 text-white' : 'bg-white/5 text-white/20 cursor-not-allowed'
            }`}>{busy ? '...' : 'Save'}</button>
        </div>
        {error && <div className="text-[9px] text-red-400">{error}</div>}
      </>)}
    </div>
  );

  if (fullscreen) {
    return (
      <div className="w-full h-full flex items-center justify-center p-4">
        <div className="bg-[#0a0f1a] border border-teal-500/30 rounded-lg max-w-lg w-full p-6">
          {content}
        </div>
      </div>
    );
  }

  return content;
};

const downloadBlob = async (url, filename) => {
  const sep = url.includes('?') ? '&' : '?';
  const authedUrl = `${url}${sep}${_authQuery()}`;
  const a = document.createElement('a');
  a.href = authedUrl;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
};

// Direct backend URL to avoid connection pool exhaustion from MJPEG streams on the proxy
const _directApi = `http://${window.location.hostname}:9010/api`;
const _directAuthFetch = (url, opts = {}) => {
  const sep = url.includes('?') ? '&' : '?';
  return fetch(`${url}${sep}${_authQuery()}`, opts);
};

const FiveMinReportPanel = ({ onGenerate, busy }) => {
  const [hh, setHh] = useState('');
  const [mm, setMm] = useState('00');
  const [ss, setSs] = useState('00');
  const [period, setPeriod] = useState('AM');
  const [date, setDate] = useState(() => new Date().toISOString().split('T')[0]);
  const mmRef = useRef(null);
  const ssRef = useRef(null);

  // Clamp and auto-advance for each field
  const handleHH = (val) => {
    const clean = val.replace(/\D/g, '').slice(0, 2);
    if (clean === '' || clean === '0') { setHh(clean); return; }
    const n = parseInt(clean, 10);
    if (n >= 1 && n <= 12) setHh(String(n));
  };
  const handleMM = (val) => {
    const clean = val.replace(/\D/g, '').slice(0, 2);
    const n = parseInt(clean, 10);
    if (clean === '') { setMm(''); return; }
    if (n >= 0 && n <= 59) setMm(clean);
  };
  const handleSS = (val) => {
    const clean = val.replace(/\D/g, '').slice(0, 2);
    const n = parseInt(clean, 10);
    if (clean === '') { setSs(''); return; }
    if (n >= 0 && n <= 59) setSs(clean);
  };
  // Pad on blur
  const padOnBlur = (val, setter) => {
    if (val === '') return;
    setter(String(parseInt(val, 10) || 0).padStart(2, '0'));
  };

  const isValid = hh !== '' && parseInt(hh, 10) >= 1 && parseInt(hh, 10) <= 12;

  const handleGenerate = () => {
    if (!isValid) return;
    const startStr = `${parseInt(hh, 10)}:${(mm || '00').padStart(2,'0')}:${(ss || '00').padStart(2,'0')}`;
    onGenerate(startStr, period, date);
  };

  const inputCls = "bg-white/5 border border-white/10 text-white/70 text-[11px] rounded px-1 py-1.5 outline-none w-8 text-center focus:border-cyan-500/50";

  return (
    <div className="p-3 space-y-2.5">
      <div className="flex items-center gap-2">
        <span className="text-[8px] text-white/40 w-8">Date</span>
        <input type="date" value={date} onChange={(e) => setDate(e.target.value)}
          className="bg-white/5 border border-white/10 text-white/70 text-[9px] rounded px-2 py-1.5 outline-none flex-1" />
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[8px] text-white/40 w-8">Time</span>
        <input value={hh} onChange={(e) => handleHH(e.target.value)} placeholder="HH"
          onBlur={() => { if (hh && parseInt(hh,10) >= 1) setHh(String(parseInt(hh,10))); }}
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          className={inputCls} />
        <span className="text-white/40 text-[11px] font-bold">:</span>
        <input ref={mmRef} value={mm} onChange={(e) => handleMM(e.target.value)} placeholder="MM"
          onFocus={(e) => e.target.select()}
          onBlur={() => padOnBlur(mm, setMm)}
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          className={inputCls} />
        <span className="text-white/40 text-[11px] font-bold">:</span>
        <input ref={ssRef} value={ss} onChange={(e) => handleSS(e.target.value)} placeholder="SS"
          onFocus={(e) => e.target.select()}
          onBlur={() => padOnBlur(ss, setSs)}
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
          className={inputCls} />
        <select value={period} onChange={(e) => setPeriod(e.target.value)}
          className="bg-white/5 border border-white/10 text-white/70 text-[9px] rounded px-1.5 py-1.5 outline-none">
          <option value="AM">AM</option>
          <option value="PM">PM</option>
        </select>
      </div>
      <button disabled={busy || !isValid} onClick={handleGenerate}
        className="w-full px-3 py-2 bg-cyan-500/10 border border-cyan-500/30 hover:bg-cyan-500/20 transition-all rounded text-[10px] text-cyan-300 font-bold disabled:opacity-40">
        {busy ? 'Generating...' : 'Generate 5-Min Report'}
      </button>
    </div>
  );
};

const HourlyReportsPanel = ({ onDownload, busy }) => {
  const [slots, setSlots] = useState([]);
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchSlots = async (date) => {
    setLoading(true);
    try {
      const res = await _directAuthFetch(`${API_BASE_URL}/crowd-live/hourly/slots?date=${date}`);
      if (res.ok) {
        const data = await res.json();
        setSlots(data.slots || []);
      } else { setSlots([]); }
    } catch (e) { setSlots([]); }
    setLoading(false);
  };

  useEffect(() => {
    (async () => {
      try {
        const res = await _directAuthFetch(`${API_BASE_URL}/crowd-live/hourly/dates`);
        if (!res.ok) { setLoading(false); return; }
        const data = await res.json();
        const d = data.dates || [];
        setDates(d);
        if (d.length > 0) {
          const latest = d[d.length - 1];
          setSelectedDate(latest);
          const sRes = await _directAuthFetch(`${API_BASE_URL}/crowd-live/hourly/slots?date=${latest}`);
          if (sRes.ok) {
            const sData = await sRes.json();
            setSlots(sData.slots || []);
          }
        }
      } catch (e) {}
      setLoading(false);
    })();
  }, []);

  if (loading) return <div className="px-4 py-3 text-[9px] text-white/40">Loading...</div>;
  if (dates.length === 0) return <div className="px-4 py-3 text-[9px] text-white/40">No hourly data yet. Start monitoring to collect data.</div>;

  return (
    <div className="max-h-64 overflow-y-auto">
      <div className="px-4 py-2 flex items-center gap-2">
        <span className="text-[8px] text-white/40 uppercase">Date:</span>
        <select
          value={selectedDate}
          onChange={(e) => { setSelectedDate(e.target.value); fetchSlots(e.target.value); }}
          className="bg-white/5 border border-white/10 text-white/80 text-[10px] rounded px-2 py-1 outline-none"
        >
          {dates.map(d => <option key={d} value={d}>{d}</option>)}
        </select>
      </div>
      {slots.length === 0 ? (
        <div className="px-4 py-2 text-[9px] text-white/40">No slots for this date.</div>
      ) : (
        slots.map(s => (
          <button
            key={s.slot}
            disabled={busy}
            onClick={() => onDownload(s.slot, selectedDate)}
            className="w-full text-left px-4 py-2 hover:bg-white/5 transition-colors flex items-center justify-between gap-2 border-b border-white/5 last:border-0"
          >
            <div className="flex items-center gap-2">
              <svg className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 2v5l3 2" /></svg>
              <div>
                <div className="text-[10px] text-white/80 font-bold">{s.slot.replace('-', ':00 - ')}:00</div>
                <div className="text-[8px] text-white/30">{s.segments} segments | {s.cameras} cameras</div>
              </div>
            </div>
            <div className="text-right flex-shrink-0">
              <div className="text-[9px] text-cyan-400 font-mono">avg {s.avg_count}</div>
              <div className="text-[8px] text-white/30">peak {s.peak_count}</div>
            </div>
          </button>
        ))
      )}
    </div>
  );
};

const ReportsDropdown = () => {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState('reports');
  const ref = useRef(null);

  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const run = async (fn) => {
    setBusy(true);
    try { await fn(); } catch (_) {}
    setBusy(false);
  };

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        disabled={busy}
        className="px-3 py-1 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/40 text-teal-400 text-[9px] font-bold uppercase tracking-wider rounded transition-colors flex items-center gap-1.5"
      >
        {busy ? (
          <div className="w-3 h-3 border border-teal-400/40 border-t-teal-400 rounded-full animate-spin" />
        ) : (
          <svg className="w-3 h-3" viewBox="0 0 16 16" fill="none"><path d="M4 8h8M4 4h8M4 12h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
        )}
        Reports
        <svg className={`w-2.5 h-2.5 transition-transform ${open ? 'rotate-180' : ''}`} viewBox="0 0 10 10" fill="none"><path d="M2 4l3 3 3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-72 bg-[#0a0f1a] border border-white/10 rounded shadow-xl z-50">
          {/* Tab switcher */}
          <div className="flex border-b border-white/10">
            <button
              onClick={() => setTab('reports')}
              className={`flex-1 py-2 text-[9px] font-bold uppercase tracking-wider transition-colors ${tab === 'reports' ? 'text-teal-400 border-b-2 border-teal-400' : 'text-white/40 hover:text-white/60'}`}
            >Reports</button>
            <button
              onClick={() => setTab('5min')}
              className={`flex-1 py-2 text-[9px] font-bold uppercase tracking-wider transition-colors ${tab === '5min' ? 'text-cyan-400 border-b-2 border-cyan-400' : 'text-white/40 hover:text-white/60'}`}
            >5-Min Report</button>
            <button
              onClick={() => setTab('hourly')}
              className={`flex-1 py-2 text-[9px] font-bold uppercase tracking-wider transition-colors ${tab === 'hourly' ? 'text-purple-400 border-b-2 border-purple-400' : 'text-white/40 hover:text-white/60'}`}
            >Hourly</button>
          </div>

          {tab === 'reports' ? (
            <div className="py-1">
              <button
                onClick={() => run(() => downloadBlob(`${API_BASE_URL}/crowd-live/report/pdf`, 'IRIS_Crowd_Live_Report.pdf'))}
                className="w-full text-left px-4 py-2.5 hover:bg-white/5 transition-colors flex items-center gap-2"
              >
                <svg className="w-3.5 h-3.5 text-red-400" viewBox="0 0 16 16" fill="currentColor"><path d="M4 1a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V5l-4-4H4zm5 0v4h4"/></svg>
                <div>
                  <div className="text-[10px] text-white/80 font-bold">Segment Report</div>
                  <div className="text-[8px] text-white/30">Current segment PDF</div>
                </div>
              </button>
              <button
                onClick={() => run(() => downloadBlob(`${API_BASE_URL}/crowd-live/master-report/pdf`, 'IRIS_Crowd_Master_Report.pdf'))}
                className="w-full text-left px-4 py-2.5 hover:bg-white/5 transition-colors flex items-center gap-2"
              >
                <svg className="w-3.5 h-3.5 text-orange-400" viewBox="0 0 16 16" fill="currentColor"><path d="M4 1a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V5l-4-4H4zm5 0v4h4"/></svg>
                <div>
                  <div className="text-[10px] text-white/80 font-bold">Master Report</div>
                  <div className="text-[8px] text-white/30">All segments PDF</div>
                </div>
              </button>
              <div className="border-t border-white/5 my-1" />
              <button
                onClick={() => run(() => downloadBlob(`${API_BASE_URL}/crowd-live/csv`, `crowd_events_${new Date().toISOString().split('T')[0]}.csv`))}
                className="w-full text-left px-4 py-2.5 hover:bg-white/5 transition-colors flex items-center gap-2"
              >
                <svg className="w-3.5 h-3.5 text-emerald-400" viewBox="0 0 16 16" fill="currentColor"><path d="M4 1a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V5l-4-4H4zm5 0v4h4"/></svg>
                <div>
                  <div className="text-[10px] text-white/80 font-bold">Export CSV</div>
                  <div className="text-[8px] text-white/30">Event log spreadsheet</div>
                </div>
              </button>
            </div>
          ) : tab === '5min' ? (
            <FiveMinReportPanel busy={busy} onGenerate={(startStr, period, date) => run(() => downloadBlob(
              `${API_BASE_URL}/crowd-live/5min/report?start=${encodeURIComponent(startStr)}&period=${period}&date=${date}`,
              `IRIS_Crowd_5min_${date}_${startStr.replace(/:/g,'')}_${period}.pdf`
            ))} />
          ) : (
            <HourlyReportsPanel
              busy={busy}
              onDownload={(slot, date) => run(() => downloadBlob(
                `${API_BASE_URL}/crowd-live/hourly/report/${slot}?date=${date}`,
                `IRIS_Crowd_Hourly_${date}_${slot}.pdf`
              ))}
            />
          )}
        </div>
      )}
    </div>
  );
};

/* ════════════════════════════════════════════════════════
   Main Dashboard
   - Top header: live badge, update time, reports dropdown, stop button
   - Left sidebar: camera list (click to switch)
   - Main area: full-screen single camera view
   ════════════════════════════════════════════════════════ */
const CrowdLiveDashboard = () => {
  const [phase, setPhase] = useState('loading');
  const [selectedCameraIds, setSelectedCameraIds] = useState([]);
  const [activeCameraId, setActiveCameraId] = useState(null);
  const [analysis, setAnalysis] = useState({});
  const [lastUpdate, setLastUpdate] = useState(null);
  const [crowdEvents, setCrowdEvents] = useState([]);
  const [sessionStartedAt, setSessionStartedAt] = useState(null);
  const [timelineVisible, setTimelineVisible] = useState(true);
  const [selectedCrowdEvent, setSelectedCrowdEvent] = useState(null);
  const [crowdEventDrawerOpen, setCrowdEventDrawerOpen] = useState(false);
  const [now, setNow] = useState(Date.now());
  const [showAddCamera, setShowAddCamera] = useState(false);
  const countHistoryRef = useRef({}); // { cameraId: [count, count, ...] }

  // On mount: check if crowd-worker is sending data → show dashboard; else → show empty
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await authFetch(`${API_BASE_URL}/crowd/analysis/latest`);
        if (res.ok && alive) {
          const raw = await res.json();
          const transformed = transformWorkerData(raw);
          if (Object.keys(transformed).length > 0) {
            setAnalysis(transformed);
            const cams = Object.keys(transformed);
            setSelectedCameraIds(cams);
            setActiveCameraId(cams[0]);
            setSessionStartedAt(new Date().toISOString());
            setPhase('running');
            return;
          }
        }
      } catch (_) {}
      if (alive) setPhase('empty');
    })();
    return () => { alive = false; };
  }, []);

  // Poll crowd-worker analysis every 5s when running
  useEffect(() => {
    if (phase !== 'running') return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await authFetch(`${API_BASE_URL}/crowd/analysis/latest`);
        if (res.ok && alive) {
          const raw = await res.json();
          const transformed = transformWorkerData(raw);
          // Accumulate count history per camera (keep last 60 points = ~5 min at 5s intervals)
          for (const [cid, d] of Object.entries(transformed)) {
            if (!countHistoryRef.current[cid]) countHistoryRef.current[cid] = [];
            countHistoryRef.current[cid].push(d.count);
            if (countHistoryRef.current[cid].length > 60) countHistoryRef.current[cid].shift();
            d.counts_history = [...countHistoryRef.current[cid]];
          }
          setAnalysis(transformed);
          // Update camera list if new cameras appear
          const newIds = Object.keys(transformed);
          setSelectedCameraIds(prev => {
            const prevSet = new Set(prev);
            const changed = newIds.some(id => !prevSet.has(id)) || prev.some(id => !transformed[id]);
            return changed ? newIds : prev;
          });
          setLastUpdate(new Date().toLocaleTimeString());
        }
      } catch (_) {}
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => { alive = false; clearInterval(t); };
  }, [phase]);

  // Auto-switch to a camera that has data if active camera has none after 15s
  useEffect(() => {
    if (phase !== 'running' || !activeCameraId) return;
    if (analysis[activeCameraId]) return; // active camera has data, all good
    // Find first camera that has analysis data
    const availableIds = Object.keys(analysis);
    if (availableIds.length > 0) {
      setActiveCameraId(availableIds[0]);
    }
  }, [phase, activeCameraId, analysis]);

  const handleCameraChanged = (data) => {
    const cams = (data.cameras || []).map(c => c.id);
    setSelectedCameraIds(cams);
    // Remove analysis for cameras no longer in the list
    setAnalysis(prev => {
      const camSet = new Set(cams);
      const next = {};
      for (const [k, v] of Object.entries(prev)) {
        if (camSet.has(k)) next[k] = v;
      }
      return next;
    });
  };

  const handleStop = async () => {
    setAnalysis({});
    setSelectedCameraIds([]);
    setActiveCameraId(null);
    setCrowdEvents([]);
    setSessionStartedAt(null);
    setSelectedCrowdEvent(null);
    setCrowdEventDrawerOpen(false);
    setPhase('empty');
  };

  // Refresh `now` every 15s for reactive timeline
  useEffect(() => {
    if (phase !== 'running') return;
    const t = setInterval(() => setNow(Date.now()), 15000);
    return () => clearInterval(t);
  }, [phase]);

  // Build timeline events from crowd-worker analysis snapshots
  useEffect(() => {
    if (phase !== 'running') return;
    // Each poll cycle, push current analysis as a timeline event
    const entries = Object.entries(analysis);
    if (entries.length === 0) return;
    const newEvents = entries.map(([cid, d]) => ({
      timestamp: new Date().toISOString(),
      segment_index: cid.replace('camera_', '').slice(0, 10),
      camera_name: d.camera_name,
      total_count: d.count,
      avg_count: d.count,
      overall_risk: d.overall_risk || 'LOW',
      crowd_density: d.condition,
    }));
    setCrowdEvents(prev => {
      const combined = [...prev, ...newEvents];
      return combined.slice(-200); // keep last 200
    });
  }, [analysis, phase]);

  // Legacy crowd events poll (kept for backward compat, will 404 gracefully)
  useEffect(() => {
    if (phase !== 'running') return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await authFetch(`${API_BASE_URL}/crowd-live/events`);
        if (res.ok && alive) {
          const data = await res.json();
          const evts = Array.isArray(data) ? data : data.events || [];
          if (evts.length > 0) setCrowdEvents(evts);
        }
      } catch (_) {}
    };
    poll();
    const t = setInterval(poll, 10000);
    return () => { alive = false; clearInterval(t); };
  }, [phase]);

  // Timeline helpers — span from first event to now
  const parseEvTime = (ev) => {
    // Try timestamp, then segment_end, then segment_start
    for (const k of ['timestamp', 'segment_end', 'segment_start']) {
      const v = ev[k];
      if (v) {
        const t = new Date(v).getTime();
        if (!isNaN(t)) return t;
      }
    }
    return null;
  };
  const getTimelineRange = () => {
    if (!crowdEvents.length) return { startMs: 0, endMs: 0, spanSec: 0 };
    let earliest = Infinity, latest = -Infinity;
    for (const ev of crowdEvents) {
      const t = parseEvTime(ev);
      if (t !== null) {
        if (t < earliest) earliest = t;
        if (t > latest) latest = t;
      }
    }
    if (earliest === Infinity) return { startMs: 0, endMs: 0, spanSec: 0 };
    // Use session start if earlier than first event
    if (sessionStartedAt) {
      const sMs = new Date(sessionStartedAt).getTime();
      if (!isNaN(sMs) && sMs < earliest) earliest = sMs;
    }
    // Extend end to now if session is running
    const nowMs = now;
    if (nowMs > latest) latest = nowMs;
    return { startMs: earliest, endMs: latest, spanSec: Math.max(1, (latest - earliest) / 1000) };
  };
  const getEventPct = (ev, range) => {
    const t = parseEvTime(ev);
    if (t === null || range.spanSec <= 0) return null;
    return Math.max(0, Math.min(100, ((t - range.startMs) / 1000 / range.spanSec) * 100));
  };
  const getElapsedSec = () => {
    const r = getTimelineRange();
    return r.spanSec;
  };
  const getEventElapsedSec = (ev) => {
    const r = getTimelineRange();
    const t = parseEvTime(ev);
    if (t === null) return null;
    return Math.max(0, (t - r.startMs) / 1000);
  };
  const fmtElapsed = (s) => {
    s = Math.round(s);
    if (s < 0) s = 0;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}` : `${m}:${String(sec).padStart(2, '0')}`;
  };
  const riskToColor = (risk) => {
    const r = String(risk || '').toUpperCase();
    if (r === 'HIGH' || r === 'CRITICAL') return 'bg-red-400';
    if (r === 'MEDIUM') return 'bg-amber-400';
    return 'bg-emerald-400';
  };

  if (phase === 'loading') {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-teal-500/30 border-t-teal-500 rounded-full animate-spin mx-auto mb-3" />
          <div className="text-[10px] text-white/30 uppercase tracking-widest">Loading...</div>
        </div>
      </div>
    );
  }

  if (phase === 'empty') {
    return <CameraManager fullscreen onChanged={(data) => {
      const cams = (data.cameras || []).map(c => c.id);
      if (cams.length > 0) {
        setSelectedCameraIds(cams);
        setActiveCameraId(cams[0] || null);
        setSessionStartedAt(new Date().toISOString());
        setPhase('running');
      }
    }} />;
  }

  // Build display list — most recently updated camera first
  const displayCameraIds = Object.keys(analysis)
    .sort((a, b) => {
      const tA = analysis[a]?.timestamp ? new Date(analysis[a].timestamp).getTime() : 0;
      const tB = analysis[b]?.timestamp ? new Date(analysis[b].timestamp).getTime() : 0;
      return tB - tA;
    });
  const activeData = activeCameraId ? analysis[activeCameraId] : null;

  return (
    <div className="w-full h-full flex flex-col">
      {/* ── TOP HEADER BAR ── */}
      <div className="flex-shrink-0 h-10 bg-[#0a0f1a] border-b border-white/10 flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          <span className="text-[10px] text-teal-400 font-black uppercase tracking-widest">Live Crowd Analysis</span>
          <span className="text-[9px] text-white/20">{displayCameraIds.length}/{selectedCameraIds.length} live</span>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdate && (
            <span className="text-[9px] text-white/30 font-mono">Updated: {lastUpdate}</span>
          )}
          <ReportsDropdown />
          <div className="relative">
            <button
              onClick={() => setShowAddCamera(v => !v)}
              className="px-3 py-1 bg-teal-500/20 hover:bg-teal-500/30 border border-teal-500/40 text-teal-400 text-[9px] font-bold uppercase tracking-wider rounded transition-colors"
            >
              Cameras
            </button>
            {showAddCamera && (
              <div className="absolute top-full right-0 mt-2 w-96 z-50 bg-[#0c1220] border border-teal-500/30 rounded-lg shadow-2xl shadow-black/60 p-4">
                <CameraManager onChanged={handleCameraChanged} />
              </div>
            )}
          </div>
          <button
            onClick={handleStop}
            className="px-3 py-1 bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-400 text-[9px] font-bold uppercase tracking-wider rounded transition-colors"
          >
            Stop
          </button>
        </div>
      </div>

      {/* ── BODY: left sidebar + main view ── */}
      <div className="relative flex flex-1 min-h-0">
        {/* LEFT SIDEBAR — camera thumbnail grid */}
        <div className="w-56 flex-shrink-0 bg-[#060a12] border-r border-white/10 overflow-y-auto">
          <div className="p-2 space-y-2">
            <div className="text-[8px] text-white/30 font-bold uppercase tracking-widest px-2 py-1">Cameras</div>
            <div className="grid grid-cols-1 gap-2">
              {displayCameraIds.map(cid => {
                const d = analysis[cid];
                const name = d?.camera_name || cid.replace('camera_', '').slice(0, 8);
                const isActive = cid === activeCameraId;
                const cond = d?.condition;
                const condColor = cond ? cc(cond) : null;
                const rawThumb = d?.heatmap_url
                  ? `${API_BASE_URL}/heatmaps/${d.heatmap_url.split('/heatmaps/').pop()}`
                  : '';
                if (rawThumb) _lastHeatmapUrls[cid] = rawThumb;
                const thumbUrl = rawThumb || _lastHeatmapUrls[cid] || '';

                return (
                  <button
                    key={cid}
                    onClick={() => setActiveCameraId(cid)}
                    className={`w-full rounded-lg overflow-hidden transition-all ${
                      isActive
                        ? 'ring-2 ring-teal-500 ring-offset-1 ring-offset-[#060a12]'
                        : 'ring-1 ring-white/10 hover:ring-white/25'
                    }`}
                  >
                    {/* thumbnail */}
                    <div className="relative w-full aspect-video bg-black">
                      <img
                        src={thumbUrl}
                        alt={name}
                        className="w-full h-full object-cover"
                      />
                      {/* dark gradient overlay at bottom */}
                      <div className="absolute inset-x-0 bottom-0 h-2/3 bg-gradient-to-t from-black/80 to-transparent pointer-events-none" />
                      {/* live dot */}
                      {d && (
                        <div className="absolute top-1.5 right-1.5 z-10">
                          <div className={`w-2 h-2 rounded-full ${condColor?.bg || 'bg-green-500'} shadow-lg`} />
                        </div>
                      )}
                      {/* camera name + stats overlay */}
                      <div className="absolute bottom-0 inset-x-0 px-2 pb-1.5 z-10">
                        <div className={`text-[9px] font-bold uppercase tracking-wider truncate ${
                          isActive ? 'text-teal-400' : 'text-white/80'
                        }`}>
                          {name}
                        </div>
                        {d && (
                          <div className="flex items-center gap-2 mt-0.5">
                            <span className="text-[8px] text-white/50">{d.count} ppl</span>
                            <span className={`text-[8px] font-bold ${condColor?.text || 'text-white/40'}`}>{cond}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* MAIN VIEW — camera OR event detail + timeline */}
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="flex-1 min-h-0 overflow-hidden">
          {selectedCrowdEvent ? (() => {
            const ev = selectedCrowdEvent;
            const risk = String(ev.overall_risk || '').toUpperCase();
            const riskColor = (risk === 'HIGH' || risk === 'CRITICAL') ? 'text-red-400' : risk === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400';
            const riskBg = (risk === 'HIGH' || risk === 'CRITICAL') ? 'bg-red-500/15 border-red-500/30' : risk === 'MEDIUM' ? 'bg-amber-500/15 border-amber-500/30' : 'bg-emerald-500/15 border-emerald-500/30';
            return (
              <div className="flex-1 overflow-y-auto bg-[#060a12] p-5">
                {/* Back button */}
                <button
                  onClick={() => { setSelectedCrowdEvent(null); setCrowdEventDrawerOpen(false); }}
                  className="mb-4 flex items-center gap-1.5 text-[9px] text-cyan-400 hover:text-cyan-300 uppercase tracking-wider font-bold transition-colors"
                >
                  <span className="text-sm">←</span> Back to camera view
                </button>

                {/* Header row */}
                <div className="flex items-center gap-3 mb-4">
                  <span className={`px-2.5 py-1 text-[9px] font-black uppercase tracking-wider rounded border ${riskBg} ${riskColor}`}>
                    {ev.overall_risk || 'N/A'}
                  </span>
                  <h2 className="text-white text-sm font-bold">
                    Segment {ev.segment_index || '--'}
                  </h2>
                  {ev.timestamp && (
                    <span className="text-[9px] text-white/30 tabular-nums">
                      @ {fmtElapsed(getEventElapsedSec(ev) || 0)}
                    </span>
                  )}
                </div>

                {/* Stats grid */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Total Count', value: ev.total_count ?? '--', big: true },
                    { label: 'Average', value: ev.avg_count ?? '--', big: true },
                    { label: 'Peak', value: ev.peak_count ?? '--', big: true },
                    { label: 'Predicted Next', value: ev.predicted_next || '--', accent: true },
                  ].map((item, i) => (
                    <div key={i} className="border border-white/10 bg-white/[0.03] rounded px-3 py-2.5">
                      <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">{item.label}</div>
                      <div className={`text-lg font-black tabular-nums ${item.accent ? 'text-cyan-400' : 'text-white/90'}`}>{item.value}</div>
                    </div>
                  ))}
                </div>

                {/* Analysis row */}
                <div className="grid grid-cols-4 gap-3 mb-4">
                  {[
                    { label: 'Density', value: ev.crowd_density },
                    { label: 'Movement', value: ev.crowd_movement },
                    { label: 'Sentiment', value: ev.sentiment },
                    { label: 'Visibility', value: ev.visibility_score },
                  ].map((item, i) => (
                    <div key={i} className="border border-white/10 bg-white/[0.03] rounded px-3 py-2">
                      <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">{item.label}</div>
                      <div className="text-[11px] text-white/80 font-medium">{item.value || '--'}</div>
                    </div>
                  ))}
                </div>

                {/* Security flags */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                  {[
                    { label: 'Weapon Detected', value: ev.weapon_detected },
                    { label: 'Fight / Collision / Injury', value: ev.fight_collision_injury },
                    { label: 'Wrongful Activity', value: ev.wrongful_activity },
                  ].map((item, i) => {
                    const isYes = String(item.value || '').toLowerCase() === 'yes';
                    return (
                      <div key={i} className={`border rounded px-3 py-2 ${isYes ? 'border-red-500/30 bg-red-500/10' : 'border-white/10 bg-white/[0.03]'}`}>
                        <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">{item.label}</div>
                        <div className={`text-[11px] font-bold ${isYes ? 'text-red-400' : 'text-emerald-400'}`}>{item.value || '--'}</div>
                      </div>
                    );
                  })}
                </div>

                {/* Time range */}
                <div className="grid grid-cols-3 gap-3 mb-4">
                  <div className="border border-white/10 bg-white/[0.03] rounded px-3 py-2">
                    <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">Segment Start</div>
                    <div className="text-[10px] text-white/60 tabular-nums">{ev.segment_start || '--'}</div>
                  </div>
                  <div className="border border-white/10 bg-white/[0.03] rounded px-3 py-2">
                    <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">Segment End</div>
                    <div className="text-[10px] text-white/60 tabular-nums">{ev.segment_end || '--'}</div>
                  </div>
                  <div className="border border-white/10 bg-white/[0.03] rounded px-3 py-2">
                    <div className="text-[8px] text-white/30 uppercase tracking-wider font-bold mb-1">Cameras Active</div>
                    <div className="text-[10px] text-white/80 tabular-nums">{ev.cameras_active || '--'}</div>
                  </div>
                </div>

                {/* Safety precaution */}
                {ev.safety_precaution && (
                  <div className="border border-amber-500/20 bg-amber-500/5 rounded px-4 py-3">
                    <div className="text-[8px] text-amber-400/70 uppercase tracking-wider font-bold mb-1">Safety Precaution</div>
                    <div className="text-[10px] text-white/70 leading-relaxed">{ev.safety_precaution}</div>
                  </div>
                )}
              </div>
            );
          })() : activeData ? (
            <FullCameraView cameraId={activeCameraId} data={activeData} />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-black">
              <div className="text-center">
                {displayCameraIds.length === 0 ? (
                  <>
                    <div className="w-8 h-8 border-2 border-teal-500/30 border-t-teal-500 rounded-full animate-spin mx-auto mb-3" />
                    <div className="text-[10px] text-white/30 uppercase tracking-widest">Connecting to cameras...</div>
                  </>
                ) : (
                  <div className="text-[10px] text-white/20 uppercase tracking-widest">Select a camera from the sidebar</div>
                )}
              </div>
            </div>
          )}
          </div>

          {/* ── SESSION TIMELINE ── */}
          {(() => {
            if (!crowdEvents.length) return null;
            const range = getTimelineRange();
            if (range.spanSec <= 0) return null;
            const startLabel = sessionStartedAt
              ? new Date(sessionStartedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
              : '0:00';
            const endLabel = new Date(range.endMs).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            return (
              <div className="flex-shrink-0 relative border-t border-cyan-500/30 bg-[#070b14] py-1.5" style={{ marginRight: activeData ? '320px' : '0' }}>
            <div className="flex items-center justify-between px-3 pb-1 text-[8px] gap-3">
              <span className="text-cyan-400/60 font-bold uppercase tracking-widest whitespace-nowrap shrink-0">Session Timeline · {crowdEvents.length} event{crowdEvents.length !== 1 ? 's' : ''}</span>
              {selectedCrowdEvent && (
                <span className="text-white/50 truncate min-w-0 text-[8px]">
                  <span className="text-white/70 font-semibold">{selectedCrowdEvent.segment_index || ''}. {(selectedCrowdEvent.crowd_density || selectedCrowdEvent.overall_risk || 'Event')}</span>
                  <span className="mx-1.5 text-white/30">·</span>
                  <span className="tabular-nums">{selectedCrowdEvent.timestamp ? new Date(selectedCrowdEvent.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : ''}</span>
                  <span className="mx-1.5 text-white/30">·</span>
                  <span>count: {selectedCrowdEvent.total_count || selectedCrowdEvent.avg_count || '?'}</span>
                  <span className="mx-1.5 text-white/30">·</span>
                  <span className={`${riskToColor(selectedCrowdEvent.overall_risk).replace('bg-', 'text-')}`}>{selectedCrowdEvent.overall_risk || 'N/A'}</span>
                </span>
              )}
              <button
                onClick={() => setTimelineVisible((v) => !v)}
                className="px-1.5 py-0.5 border border-cyan-500/35 text-cyan-300 hover:border-cyan-300/70 hover:text-cyan-200 uppercase tracking-wider shrink-0"
              >
                {timelineVisible ? 'Hide' : 'Show'}
              </button>
            </div>
            {timelineVisible && (
              <>
                <div
                  className="relative mx-3 h-7 border border-white/10 bg-black/50 overflow-visible cursor-pointer rounded-sm"
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                    const clickPct = (x / rect.width) * 100;
                    let best = null;
                    let bestDist = Infinity;
                    for (const ev of crowdEvents) {
                      const pct = getEventPct(ev, range);
                      if (pct === null) continue;
                      const d = Math.abs(pct - clickPct);
                      if (d < bestDist) { bestDist = d; best = ev; }
                    }
                    if (best && bestDist <= 3) {
                      setSelectedCrowdEvent(best);
                    }
                  }}
                >
                  <div className="absolute inset-y-0 left-0 right-0 bg-white/5" />
                  {range.spanSec > 0 && Array.from({ length: Math.min(200, Math.floor(range.spanSec / 60) + 1) }).map((_, i) => {
                    const leftPct = Math.max(0, Math.min(100, (i * 60 / range.spanSec) * 100));
                    const isMajor = i % 5 === 0;
                    return (
                      <div
                        key={`tick-${i}`}
                        className={`absolute bottom-0 w-px ${isMajor ? 'bg-white/25 h-5' : 'bg-white/10 h-3'}`}
                        style={{ left: `${leftPct}%` }}
                      />
                    );
                  })}
                  {crowdEvents.map((ev, i) => {
                    const leftPct = getEventPct(ev, range);
                    if (leftPct === null) return null;
                    const isSelected = selectedCrowdEvent && selectedCrowdEvent.timestamp === ev.timestamp && selectedCrowdEvent.segment_index === ev.segment_index;
                    return (
                      <button
                        key={`ev-${i}`}
                        title={`Seg ${ev.segment_index || i + 1} · ${ev.overall_risk || 'N/A'}`}
                        onClick={(evClick) => {
                          evClick.stopPropagation();
                          setSelectedCrowdEvent(ev);
                        }}
                        className={`absolute top-0 -translate-x-1/2 w-[3px] h-full ${riskToColor(ev.overall_risk)} hover:brightness-125 transition-all ${isSelected ? 'ring-1 ring-white z-10 w-[5px]' : 'opacity-70 hover:opacity-100'}`}
                        style={{ left: `${leftPct}%` }}
                      />
                    );
                  })}
                </div>
                <div className="mt-1 px-3 flex items-center justify-between text-[8px] text-white/35 leading-none tabular-nums">
                  <span>{startLabel}</span>
                  <div className="flex items-center gap-3">
                    <span>{crowdEvents.length} events</span>
                    <span>{fmtElapsed(range.spanSec)}</span>
                  </div>
                  <span>{endLabel}</span>
                </div>
              </>
            )}
              </div>
            );
          })()}
        </div>
      </div>

    </div>
  );
};

export default CrowdLiveDashboard;
