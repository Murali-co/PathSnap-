import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  CircleMarker,
  LayersControl,
  MapContainer,
  Polyline,
  TileLayer,
  useMap,
} from 'react-leaflet';
import {
  ApiError,
  getHealth,
  getJob,
  getResult,
  getTrips,
  startPrediction,
  toMapPositions,
  type Health,
  type MatchResult,
} from './api';

const POLL_INTERVAL_MS = 1200;
const JOB_TIMEOUT_MS = 15 * 60 * 1000;

function MapBounds({ points }: { points: [number, number][] }) {
  const map = useMap();

  useEffect(() => {
    if (points.length > 1) {
      map.fitBounds(points, { padding: [36, 36], maxZoom: 16 });
    } else if (points.length === 1) {
      map.setView(points[0], 15);
    }
  }, [map, points]);

  return null;
}

function formatSeconds(seconds: number | undefined): string {
  return typeof seconds === 'number' && Number.isFinite(seconds)
    ? `${seconds.toFixed(2)}s`
    : '—';
}

function formatLabel(value: string | null | undefined): string {
  if (!value) return 'Unavailable';
  return value.replace(/_/g, ' ');
}

function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [trips, setTrips] = useState<string[]>([]);
  const [tripWarning, setTripWarning] = useState<string | null>(null);
  const [selectedTrip, setSelectedTrip] = useState('');
  const [result, setResult] = useState<MatchResult | null>(null);
  const [jobStatus, setJobStatus] = useState('');
  const [stage, setStage] = useState('');
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);
  const [showBasemap, setShowBasemap] = useState(false);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [healthResponse, tripsResponse] = await Promise.all([
        getHealth(),
        getTrips(),
      ]);
      setHealth(healthResponse);
      setTrips(tripsResponse.trips);
      setTripWarning(tripsResponse.warning);
      setSelectedTrip((current) =>
        current && tripsResponse.trips.includes(current)
          ? current
          : (tripsResponse.trips[0] ?? ''),
      );
    } catch (cause) {
      setHealth(null);
      setTrips([]);
      setTripWarning(null);
      setError(cause instanceof Error ? cause.message : 'Unable to load the dashboard.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard, refreshKey]);

  const points = useMemo(
    () => (result ? toMapPositions(result.geometry.gps) : []),
    [result],
  );
  const gpsSegments = useMemo(() => result?.geometry.gps_segments?.map(toMapPositions).filter((line) => line.length > 1) ?? (points.length > 1 ? [points] : []), [result, points]);
  const decodedSegments = useMemo(() => result?.geometry.predicted?.map(toMapPositions).filter((line) => line.length > 1) ?? [], [result]);
  const classifierSegments = useMemo(() => result?.geometry.classifier?.map(toMapPositions).filter((line) => line.length > 1) ?? [], [result]);
  const networkSegments = useMemo(() => result?.geometry.network?.map(toMapPositions).filter((line) => line.length > 1) ?? [], [result]);
  const arcIds = useMemo(
    () =>
      result?.geometry.predicted_arc_ids?.filter(
        (id): id is number => typeof id === 'number' && Number.isFinite(id),
      ) ?? [],
    [result],
  );

  const runMatch = async () => {
    if (!selectedTrip || busy) return;
    setBusy(true);
    setError('');
    setResult(null);
    setJobStatus('queued');
    setStage('pending');
    try {
      const started = await startPrediction(selectedTrip);
      if (!started.job_id || started.status === 'failed') {
        throw new ApiError(started.error || 'The API could not start this prediction.');
      }

      const deadline = Date.now() + JOB_TIMEOUT_MS;
      while (Date.now() < deadline) {
        const job = await getJob(started.job_id);
        setJobStatus(job.status);
        setStage(job.stage || '');

        if (job.status === 'failed') {
          throw new ApiError(job.safe_error || 'Prediction failed.');
        }
        if (job.status === 'completed') {
          setResult(await getResult(started.job_id));
          setJobStatus('completed');
          return;
        }

        await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
      }
      throw new ApiError('Prediction timed out while waiting for the backend.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Prediction failed.');
      setJobStatus('failed');
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="PathSnap home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M7 23.5 13.2 8l5.1 12.2 2.8-6.1L25 23.5" />
              <circle cx="7" cy="23.5" r="2" />
              <circle cx="25" cy="23.5" r="2" />
            </svg>
          </span>
          <span className="brand-name">path<span>snap</span></span>
        </a>
        <div className="topbar-right">
          <span className="eyebrow">MAP MATCHING WORKSPACE</span>
          <div className={`connection ${health ? 'is-online' : 'is-offline'}`}>
            <span className="status-dot" />
            {loading ? 'Connecting' : health ? 'API connected' : 'API offline'}
          </div>
        </div>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow hero-eyebrow">PATHSNAP / ROUTE ANALYSIS</div>
          <h1>Make every trace<br /><span>tell a story.</span></h1>
          <p>Inspect a GPS trip and run it through the PathSnap matching pipeline.</p>
        </div>
        <div className="hero-meta">
          <span className="meta-label">ENVIRONMENT</span>
          <strong>{health?.mode ?? '—'}</strong>
          <span className="meta-divider" />
          <span className="meta-label">MODEL</span>
          <strong>{health?.model_version ?? '—'}</strong>
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <span className="section-index">01</span>
            <div>
              <span className="eyebrow">CONFIGURATION</span>
              <h2>Choose a trip</h2>
            </div>
          </div>

          <label className="field-label" htmlFor="trip-select">TRIP ID</label>
          <div className="select-wrap">
            <select
              id="trip-select"
              value={selectedTrip}
              onChange={(event) => {
                setSelectedTrip(event.target.value);
                setResult(null);
                setError('');
              }}
              disabled={loading || trips.length === 0 || busy}
            >
              {trips.length === 0 && <option value="">No trips available</option>}
              {trips.map((trip) => <option key={trip} value={trip}>{trip}</option>)}
            </select>
            <span className="select-chevron" aria-hidden="true">⌄</span>
          </div>

          <div className="trip-count">
            <span className="count-icon" aria-hidden="true">↗</span>
            <span><strong>{trips.length}</strong> trips available</span>
            <span className="count-mode">{health?.mode ?? '—'}</span>
          </div>

          <button
            className="run-button"
            type="button"
            onClick={() => void runMatch()}
            disabled={loading || !health?.artifacts_ready || health.data_ready === false || !selectedTrip || busy}
          >
            {busy ? (
              <>
                <span className="spinner" />
                <span>Matching<span className="button-subtitle">{formatLabel(stage)}</span></span>
              </>
            ) : (
              <>
                <span>Run map match</span>
                <span className="button-arrow" aria-hidden="true">↗</span>
              </>
            )}
          </button>

          {error && (
            <div className="notice notice-error" role="alert">
              <span className="notice-symbol">!</span>
              <div><strong>Something went wrong</strong><p>{error}</p></div>
            </div>
          )}

          {tripWarning && !error && (
            <div className="notice notice-neutral">
              <span className="notice-symbol">i</span>
              <div><strong>Trip data unavailable</strong><p>{tripWarning}</p></div>
            </div>
          )}

          {!loading && !health && (
            <button
              className="retry-button"
              type="button"
              onClick={() => setRefreshKey((key) => key + 1)}
            >
              Retry connection <span aria-hidden="true">↻</span>
            </button>
          )}

          <div className="panel-footnote">
            <span className="footnote-line" />
            <p>Each run processes the selected trace and reports the result when the backend job completes.</p>
          </div>
        </aside>

        <section className="results-panel" aria-label="Trip match results">
          <div className="panel-heading results-heading">
            <span className="section-index">02</span>
            <div>
              <span className="eyebrow">LIVE OUTPUT</span>
              <h2>{result ? 'Match overview' : busy ? 'Processing trip' : 'Map canvas'}</h2>
            </div>
            <span className={`run-state ${busy ? 'state-running' : result ? 'state-done' : 'state-idle'}`}>
              <span className="status-dot" />
              {busy ? formatLabel(jobStatus) : result ? 'Completed' : 'Ready'}
            </span>
          </div>

          {result && (
            <div className="metrics-grid">
              <Metric label="OBSERVATIONS" value={String(result.observations)} detail="GPS points" />
              <Metric label="PROCESSING TIME" value={formatSeconds(result.timing.total_seconds)} detail="Total duration" />
              <Metric label="EVALUATION" value={formatLabel(result.evaluation_status)} detail="Reference status" />
              <Metric
                label="DECODED TOP-1"
                value={typeof result.metrics?.decoded?.observation_top1 === 'number' ? `${(result.metrics.decoded.observation_top1 * 100).toFixed(2)}%` : 'Unavailable'}
                detail={result.label_method === 'ordered_occurrence_dp_v1' ? 'Derived reference score' : 'Reference score'}
              />
            </div>
          )}

          <label className="geometry-note" style={{ padding: '8px 24px', display: 'block' }}>
            <input type="checkbox" checked={showBasemap} onChange={(event) => setShowBasemap(event.target.checked)} /> Online basemap (optional; needs internet)
          </label>
          <div className="map-frame">
            {points.length > 0 ? (
              <MapContainer
                key={result?.trip_id ?? 'empty-map'}
                center={points[0]}
                zoom={13}
                scrollWheelZoom
                preferCanvas
                className="leaflet-map"
              >
                {showBasemap && <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />}
                <MapBounds points={points} />
                <LayersControl position="topright">
                  <LayersControl.Overlay checked name="Road network">
                    <Polyline positions={networkSegments} pathOptions={{ color: '#94a3b8', weight: 1, opacity: 0.5 }} />
                  </LayersControl.Overlay>
                  <LayersControl.Overlay checked name="GPS observations">
                    <Polyline positions={gpsSegments} pathOptions={{ color: '#60a5fa', weight: 2, opacity: 0.85 }} />
                  </LayersControl.Overlay>
                  <LayersControl.Overlay name="Classifier route">
                    <Polyline positions={classifierSegments} pathOptions={{ color: '#f3a66b', weight: 3 }} />
                  </LayersControl.Overlay>
                  <LayersControl.Overlay checked name="Decoded route">
                    <Polyline positions={decodedSegments} pathOptions={{ color: '#b8f36b', weight: 4, opacity: 0.95 }} />
                  </LayersControl.Overlay>
                </LayersControl>
                <CircleMarker center={points[0]} radius={7} pathOptions={{ color: '#101815', weight: 3, fillColor: '#b8f36b', fillOpacity: 1 }} />
                {points.length > 1 && (
                  <CircleMarker center={points[points.length - 1]} radius={7} pathOptions={{ color: '#101815', weight: 3, fillColor: '#f3a66b', fillOpacity: 1 }} />
                )}
              </MapContainer>
            ) : (
              <div className="map-placeholder">
                <div className="map-grid" aria-hidden="true" />
                <div className="placeholder-route" aria-hidden="true">
                  <span className="route-point route-start" />
                  <span className="route-path" />
                  <span className="route-point route-end" />
                </div>
                <div className="placeholder-copy">
                  <span className="placeholder-icon" aria-hidden="true">⌖</span>
                  <strong>{busy ? 'Loading trip geometry' : result ? 'No valid GPS points' : 'Your trip map will appear here'}</strong>
                  <span>{busy ? 'Waiting for the backend to finish processing.' : 'Select a trip and run map matching to visualize its GPS trace.'}</span>
                </div>
                <span className="map-coordinates">GPS TRACE <i>—</i> LIVE VIEW</span>
              </div>
            )}
            {points.length > 0 && (
              <div className="map-overlay map-overlay-top">
                <span className="live-dot" /> ROUTE VIEW <span className="overlay-divider">/</span> {result?.trip_id}
              </div>
            )}
            {points.length > 0 && (
              <div className="map-legend">
                <span><i className="legend-line" /> Decoded route</span>
                <span style={{ color: '#60a5fa' }}>● GPS</span>
                <span><i className="legend-start" /> Start</span>
                <span><i className="legend-end" /> End</span>
              </div>
            )}
          </div>

          {result && (
            <div className="result-footer">
              <div className="result-context">
                <span className="eyebrow">MATCHING METHOD</span>
                <strong>{formatLabel(result.label_method)}</strong>
              </div>
              {arcIds.length > 0 && (
                <details className="arc-details">
                  <summary>View {arcIds.length} predicted arc IDs <span aria-hidden="true">⌄</span></summary>
                  <div className="arc-list">
                    {arcIds.map((id, index) => <span key={`${index}-${id}`}>{id}</span>)}
                  </div>
                </details>
              )}
              <p className="geometry-note">
                {decodedSegments.length > 0 ? `Decoded route from the API · ${result.diagnostics?.blocks ?? decodedSegments.length} route blocks. Gaps remain separate; intermediate arcs are predicted paths.` : 'The backend returned no decoded road geometry for this result.'}
              </p>
            </div>
          )}
        </section>
      </section>
      <footer className="page-footer">
        <span>PATHSNAP <i>·</i> GPS MATCHING</span>
        <span>Built for trace exploration</span>
      </footer>
    </main>
  );
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <article className="metric-card">
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-detail">{detail}</span>
    </article>
  );
}

export default App;
