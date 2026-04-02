const BASE = "/api";

export interface SystemInfo {
  id: string;
  label: string;
}

export interface RecommendRequest {
  model: string;
  gpus: number;
  system: string;
  isl: number;
  osl: number;
  ttft: number;
  tpot: number;
  prefix: number;
  top_n: number;
}

export interface RecommendResult {
  chosen_mode: string;
  best_throughput_tok_s: number;
  best_throughput_per_gpu: number;
  best_throughput_per_user: number;
  ttft_ms: number;
  tpot_ms: number;
  request_latency_ms: number;
  agg_configs: Record<string, any>[];
  disagg_configs: Record<string, any>[];
  model_path: string;
  system: string;
  total_gpus: number;
  isl: number;
  osl: number;
  speedup: number | null;
}

export interface SupportResult {
  model_path: string;
  system: string;
  agg_supported: boolean;
  disagg_supported: boolean;
}

export interface GenerateResult {
  model_path: string;
  system: string;
  backend: string;
  backend_version: string;
  total_gpus: number;
  tp: number;
  pp: number;
  replicas: number;
  max_batch_size: number;
  output_dir: string | null;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const fetchSystems = (): Promise<SystemInfo[]> =>
  fetch(`${BASE}/systems`).then((r) => r.json());

export const recommend = (req: RecommendRequest): Promise<RecommendResult> =>
  post("/recommend", req);

export const checkSupport = (model: string, system: string): Promise<SupportResult> =>
  post("/support", { model, system });

export const generate = (model: string, gpus: number, system: string): Promise<GenerateResult> =>
  post("/generate", { model, gpus, system });
