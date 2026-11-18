export interface Health {
  status: string;
  mode: string;
  model_version: string;
  artifacts_ready: boolean;
  database_configured?: boolean;
  data_ready?: boolean;
  error?: string | null;
}

export interface TripsResponse {
  mode: string;
  trips: string[];
  warning: string | null;
}

export interface PredictionResponse {
  job_id: string | null;
  status: string;
  error?: string;
}

export interface Job {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage?: string;
  safe_error?: string;
}

export interface MatchResult {
  job_id: string;
  status: string;
  trip_id: string;
  evaluation_status: string;
  label_method: string | null;
  observations: number;
  geometry: {
    gps: Array<number[] | null>;
    gps_segments?: number[][][];
    predicted?: number[][][];
    classifier?: number[][][];
    network?: number[][][];
    predicted_arc_ids?: Array<number | null>;
  };
  timing: {
    total_seconds: number;
  };
  predictions_available: boolean;
  metrics?: { decoded?: { observation_top1?: number | null; route_continuity?: number | null }; classifier?: { observation_top1?: number | null } } | null;
  diagnostics?: { blocks?: number; restarts?: number; unmatched_observations?: number };
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 30000);
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(controller.signal.aborted ? 'The API request timed out. Retry or check the backend.' : 'Could not reach the PathSnap API. Check the backend connection.');
  } finally {
    window.clearTimeout(timeoutId);
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      typeof body === 'object' && body !== null && 'detail' in body
        ? String(body.detail)
        : response.statusText;
    throw new ApiError(detail || `Request failed (${response.status})`, response.status);
  }

  return body as T;
}

export const getHealth = () => request<Health>('/health');

export const getTrips = () => request<TripsResponse>('/trips');

export const startPrediction = (tripId: string) =>
  request<PredictionResponse>(`/predict/${encodeURIComponent(tripId)}`, {
    method: 'POST',
  });

export const getJob = (jobId: string) =>
  request<Job>(`/jobs/${encodeURIComponent(jobId)}`);

export const getResult = (jobId: string) =>
  request<MatchResult>(`/results/${encodeURIComponent(jobId)}`);

export function toMapPositions(points: Array<number[] | null>): [number, number][] {
  return points
    .filter(
      (point): point is number[] =>
        Array.isArray(point) && point.length >= 2 &&
        Number.isFinite(point[0]) &&
        Number.isFinite(point[1]) &&
        Math.abs(point[0]) <= 180 &&
        Math.abs(point[1]) <= 90,
    )
    .map(([longitude, latitude]) => [latitude, longitude]);
}
