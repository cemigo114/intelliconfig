import { useState, useEffect, useRef, type CSSProperties } from "react";
import {
  fetchSystems,
  recommend,
  fetchManifest,
  type SystemInfo,
  type RecommendResult,
  type ManifestResult,
} from "./api";

// ── Design tokens ──

const T = {
  bg: "#0f1117",
  surface: "#161922",
  card: "#1c1f2b",
  border: "#2a2d3a",
  borderHover: "#3d4155",
  accent: "#ef4444",
  accentSoft: "rgba(239,68,68,0.12)",
  teal: "#14b8a6",
  tealSoft: "rgba(20,184,166,0.12)",
  blue: "#3b82f6",
  green: "#22c55e",
  orange: "#f59e0b",
  purple: "#a855f7",
  text: "#e2e4e9",
  textMuted: "#8b8fa3",
  textDim: "#555970",
  white: "#f8f9fa",
  mono: "'JetBrains Mono', monospace",
  sans: "'Inter', system-ui, sans-serif",
};

// ── Decision tree data ──

interface TreeOption {
  label: string;
  value: string;
  detail?: string;
}

interface TreeNode {
  id: string;
  question: string;
  hint: string;
  options: TreeOption[];
}

const MODELS: TreeOption[] = [
  { label: "Qwen/Qwen3-32B-FP8", value: "Qwen/Qwen3-32B-FP8", detail: "32B dense, FP8" },
  { label: "meta-llama/Llama-3.1-70B", value: "meta-llama/Llama-3.1-70B", detail: "70B dense" },
  { label: "meta-llama/Llama-3.1-8B", value: "meta-llama/Llama-3.1-8B", detail: "8B dense" },
  { label: "deepseek-ai/DeepSeek-V3", value: "deepseek-ai/DeepSeek-V3", detail: "MoE, MLA" },
];

const GPU_COUNTS: TreeOption[] = [
  { label: "2 GPUs", value: "2" },
  { label: "4 GPUs", value: "4" },
  { label: "8 GPUs", value: "8" },
  { label: "16 GPUs", value: "16" },
  { label: "32 GPUs", value: "32" },
];

const WORKLOADS: TreeOption[] = [
  { label: "Short context, High QPS (chatbot)", value: "short", detail: "ISL=1000, OSL=500" },
  { label: "Long prefill, RAG/summarization", value: "long-prefill", detail: "ISL=4000, OSL=1000" },
  { label: "Long context, code generation", value: "long-long", detail: "ISL=8000, OSL=2000" },
  { label: "Custom workload shape", value: "custom" },
];

const OBJECTIVES: TreeOption[] = [
  { label: "Minimize latency (TTFT < 500ms)", value: "latency", detail: "TTFT=500, TPOT=15" },
  { label: "Maximize throughput (tok/s/$)", value: "throughput", detail: "TTFT=2000, TPOT=30" },
  { label: "Balanced", value: "balanced", detail: "TTFT=1000, TPOT=20" },
];

const WORKLOAD_PARAMS: Record<string, { isl: number; osl: number }> = {
  short: { isl: 1000, osl: 500 },
  "long-prefill": { isl: 4000, osl: 1000 },
  "long-long": { isl: 8000, osl: 2000 },
  custom: { isl: 4000, osl: 1000 },
};

const OBJECTIVE_PARAMS: Record<string, { ttft: number; tpot: number }> = {
  latency: { ttft: 500, tpot: 15 },
  throughput: { ttft: 2000, tpot: 30 },
  balanced: { ttft: 1000, tpot: 20 },
};

// ── Components ──

function StepIndicator({ steps, current }: { steps: string[]; current: number }) {
  return (
    <div style={{ display: "flex", gap: 4, alignItems: "center", marginBottom: 24 }}>
      {steps.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 12,
                fontWeight: 700,
                fontFamily: T.mono,
                background: done ? T.accent : active ? T.card : T.surface,
                border: `2px solid ${done ? T.accent : active ? T.teal : T.border}`,
                color: done || active ? T.white : T.textDim,
                boxShadow: active ? `0 0 12px ${T.teal}40` : "none",
                transition: "all 0.3s",
              }}
            >
              {done ? "✓" : i + 1}
            </div>
            <span
              style={{
                fontSize: 11,
                fontFamily: T.mono,
                color: active ? T.teal : done ? T.text : T.textDim,
                fontWeight: active ? 600 : 400,
              }}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div
                style={{
                  width: 24,
                  height: 2,
                  background: done ? T.accent : T.border,
                  marginLeft: 4,
                  marginRight: 4,
                  transition: "all 0.3s",
                }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

function OptionCard({
  option,
  selected,
  onClick,
}: {
  option: TreeOption;
  selected: boolean;
  onClick: () => void;
}) {
  const [hovered, setHovered] = useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: "100%",
        padding: "12px 16px",
        background: selected ? T.accentSoft : hovered ? T.card : "transparent",
        border: `1px solid ${selected ? T.accent : hovered ? T.borderHover : "transparent"}`,
        borderLeft: `3px solid ${selected ? T.accent : hovered ? T.teal : T.border}`,
        borderRadius: 8,
        color: selected ? T.white : T.text,
        fontSize: 13,
        fontFamily: T.mono,
        cursor: "pointer",
        textAlign: "left",
        transition: "all 0.15s",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <span>{option.label}</span>
      {option.detail && (
        <span style={{ fontSize: 11, color: T.textMuted, fontWeight: 400 }}>{option.detail}</span>
      )}
    </button>
  );
}

function TreeStep({
  node,
  selected,
  onSelect,
  active,
}: {
  node: TreeNode;
  selected: string | null;
  onSelect: (value: string) => void;
  active: boolean;
}) {
  return (
    <div
      style={{
        opacity: active ? 1 : 0.35,
        pointerEvents: active ? "auto" : "none",
        transition: "opacity 0.3s",
        marginBottom: 20,
      }}
    >
      <div style={{ marginBottom: 8 }}>
        <div style={{ color: T.white, fontSize: 14, fontWeight: 600, fontFamily: T.sans }}>
          {node.question}
        </div>
        <div style={{ color: T.textMuted, fontSize: 11, marginTop: 2, fontStyle: "italic" }}>
          {node.hint}
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginLeft: 8 }}>
        {node.options.map((opt) => (
          <OptionCard
            key={opt.value}
            option={opt}
            selected={selected === opt.value}
            onClick={() => onSelect(opt.value)}
          />
        ))}
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div
      style={{
        background: T.card,
        border: `1px solid ${T.border}`,
        borderRadius: 10,
        padding: "14px 16px",
        textAlign: "center",
        flex: 1,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: T.textMuted,
          textTransform: "uppercase",
          letterSpacing: 1,
          marginBottom: 6,
          fontFamily: T.mono,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: T.sans }}>{value}</div>
    </div>
  );
}

function ConfigTable({ title, configs }: { title: string; configs: Record<string, any>[] }) {
  if (!configs.length) return null;
  const keys = Object.keys(configs[0]).filter((k) => k !== "backend").slice(0, 8);
  return (
    <div
      style={{
        background: T.card,
        border: `1px solid ${T.border}`,
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "10px 16px",
          borderBottom: `1px solid ${T.border}`,
          fontSize: 12,
          fontWeight: 600,
          color: T.text,
          fontFamily: T.sans,
        }}
      >
        {title}
      </div>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: T.mono,
            fontSize: 11,
          }}
        >
          <thead>
            <tr>
              {keys.map((k) => (
                <th
                  key={k}
                  style={{
                    padding: "8px 10px",
                    textAlign: "left",
                    color: T.textMuted,
                    borderBottom: `1px solid ${T.border}`,
                    whiteSpace: "nowrap",
                  }}
                >
                  {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {configs.slice(0, 5).map((row, i) => (
              <tr key={i}>
                {keys.map((k) => (
                  <td
                    key={k}
                    style={{
                      padding: "6px 10px",
                      color: T.text,
                      borderBottom: `1px solid ${T.border}20`,
                      whiteSpace: "nowrap",
                    }}
                  >
                    {typeof row[k] === "number" ? Number(row[k]).toFixed(2) : String(row[k] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CLIPreview({ choices }: { choices: Record<string, string | null> }) {
  const parts = ["intelliconfig recommend"];
  if (choices.model) parts.push(`\\\n  --model ${choices.model}`);
  if (choices.gpus) parts.push(`\\\n  --gpus ${choices.gpus}`);
  if (choices.system) parts.push(`\\\n  --system ${choices.system}`);
  if (choices.workload && choices.workload !== "custom") {
    const wp = WORKLOAD_PARAMS[choices.workload];
    parts.push(`\\\n  --isl ${wp.isl} --osl ${wp.osl}`);
  }
  if (choices.objective) {
    const op = OBJECTIVE_PARAMS[choices.objective];
    parts.push(`\\\n  --ttft ${op.ttft} --tpot ${op.tpot}`);
  }

  return (
    <div
      style={{
        background: "#0a0c10",
        borderRadius: 10,
        border: `1px solid ${T.border}`,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "8px 14px",
          background: "#12141a",
          borderBottom: `1px solid ${T.border}`,
        }}
      >
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#ff5f57" }} />
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#febd2e" }} />
        <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#27ca40" }} />
        <span style={{ color: T.textDim, fontSize: 11, fontFamily: T.mono, marginLeft: 8 }}>
          intelliconfig — bash
        </span>
      </div>
      <pre
        style={{
          padding: "14px 16px",
          margin: 0,
          fontFamily: T.mono,
          fontSize: 12,
          lineHeight: 1.8,
          color: T.teal,
          overflowX: "auto",
        }}
      >
        <span style={{ color: T.green }}>$ </span>
        {parts.join(" ")}
        <span style={{ color: T.accent, animation: "blink 1s step-end infinite" }}>▊</span>
      </pre>
    </div>
  );
}

// ── Deploy Panel ──

function ModeToggle({
  mode,
  hasAgg,
  hasDisagg,
  onChange,
}: {
  mode: string;
  hasAgg: boolean;
  hasDisagg: boolean;
  onChange: (m: string) => void;
}) {
  const opts = [
    { value: "agg", label: "Aggregated", enabled: hasAgg },
    { value: "disagg", label: "Disaggregated (P/D)", enabled: hasDisagg },
  ];
  return (
    <div style={{ display: "flex", gap: 6 }}>
      {opts.map((o) => {
        const active = mode === o.value || (!mode && o.value === "agg");
        return (
          <button
            key={o.value}
            disabled={!o.enabled}
            onClick={() => onChange(o.value)}
            style={{
              flex: 1,
              padding: "8px 12px",
              borderRadius: 6,
              border: `1px solid ${active ? T.teal : T.border}`,
              background: active ? T.tealSoft : "transparent",
              color: !o.enabled ? T.textDim : active ? T.teal : T.text,
              fontSize: 12,
              fontFamily: T.mono,
              fontWeight: active ? 600 : 400,
              cursor: o.enabled ? "pointer" : "not-allowed",
              opacity: o.enabled ? 1 : 0.4,
              transition: "all 0.15s",
            }}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

function InputField({
  label,
  value,
  onChange,
  placeholder,
  mono,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  mono?: boolean;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label
        style={{
          fontSize: 10,
          color: T.textMuted,
          textTransform: "uppercase",
          letterSpacing: 1,
          fontFamily: T.mono,
        }}
      >
        {label}
      </label>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          padding: "7px 10px",
          borderRadius: 6,
          border: `1px solid ${T.border}`,
          background: T.surface,
          color: T.text,
          fontSize: 12,
          fontFamily: mono ? T.mono : T.sans,
          outline: "none",
        }}
      />
    </div>
  );
}

function DeployPanel({ result }: { result: RecommendResult }) {
  const engineMode = result.chosen_mode.includes("disagg") ? "disagg" : "agg";
  const hasAgg = result.agg_configs.length > 0;
  const hasDisagg = result.disagg_configs.length > 0;

  const [mode, setMode] = useState(engineMode);
  const [namespace, setNamespace] = useState("llm-d");
  const [gatewayName, setGatewayName] = useState("inference-gateway");
  const [imageOverride, setImageOverride] = useState("");
  const [maxModelLen, setMaxModelLen] = useState("");
  const [manifest, setManifest] = useState<ManifestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    setManifest(null);
    try {
      const res = await fetchManifest({
        recommend_result: result,
        mode_override: mode !== engineMode ? mode : null,
        namespace,
        gateway_name: gatewayName,
        image: imageOverride || null,
        max_model_len: maxModelLen ? parseInt(maxModelLen, 10) : null,
      });
      setManifest(res);
    } catch (e: any) {
      setError(e.message || "Manifest generation failed");
    } finally {
      setLoading(false);
    }
  }

  function handleCopy() {
    if (!manifest) return;
    navigator.clipboard.writeText(manifest.yaml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const isDisagg = mode === "disagg";

  return (
    <div
      style={{
        background: T.card,
        border: `1px solid ${T.border}`,
        borderRadius: 10,
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 16px",
          borderBottom: `1px solid ${T.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: T.white, fontFamily: T.sans }}>
            Deploy to llm-d
          </div>
          <div style={{ fontSize: 11, color: T.textMuted, marginTop: 2 }}>
            Generate Kubernetes manifests for your cluster
          </div>
        </div>
        <div
          style={{
            padding: "3px 10px",
            borderRadius: 4,
            fontSize: 10,
            fontWeight: 700,
            fontFamily: T.mono,
            textTransform: "uppercase",
            letterSpacing: 0.5,
            background: isDisagg ? "rgba(168,85,247,0.15)" : T.tealSoft,
            color: isDisagg ? T.purple : T.teal,
            border: `1px solid ${isDisagg ? T.purple + "44" : T.teal + "44"}`,
          }}
        >
          {isDisagg ? "P/D Disaggregated" : "Aggregated"}
        </div>
      </div>

      {/* Config form */}
      <div style={{ padding: 16, display: "flex", flexDirection: "column", gap: 14 }}>
        {/* Mode selector */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label
            style={{
              fontSize: 10,
              color: T.textMuted,
              textTransform: "uppercase",
              letterSpacing: 1,
              fontFamily: T.mono,
            }}
          >
            Serving Mode
            {mode !== engineMode && (
              <span style={{ color: T.orange, marginLeft: 6, textTransform: "none", letterSpacing: 0 }}>
                (overriding engine recommendation)
              </span>
            )}
          </label>
          <ModeToggle mode={mode} hasAgg={hasAgg} hasDisagg={hasDisagg} onChange={setMode} />
          <div style={{ fontSize: 11, color: T.textDim, marginTop: 2 }}>
            {isDisagg
              ? "Separate prefill and decode workers with NIXL KV-cache transfer. Best for large models and long prompts."
              : "Single worker pool handles both prefill and decode. Simpler setup, good for most workloads."}
          </div>
        </div>

        {/* Namespace + Gateway */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <InputField
            label="Namespace"
            value={namespace}
            onChange={setNamespace}
            placeholder="llm-d"
            mono
          />
          <InputField
            label="Gateway Name"
            value={gatewayName}
            onChange={setGatewayName}
            placeholder="inference-gateway"
            mono
          />
        </div>

        {/* Advanced: image + max-model-len */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          <InputField
            label="vLLM Image (optional)"
            value={imageOverride}
            onChange={setImageOverride}
            placeholder="ghcr.io/llm-d/llm-d-cuda:v0.5.1"
            mono
          />
          <InputField
            label="Max Model Length (optional)"
            value={maxModelLen}
            onChange={setMaxModelLen}
            placeholder="e.g. 32000"
            mono
          />
        </div>

        {/* Generate button */}
        <button
          onClick={handleGenerate}
          disabled={loading}
          style={{
            padding: "10px 16px",
            borderRadius: 6,
            border: "none",
            background: T.teal,
            color: "#000",
            fontSize: 13,
            fontWeight: 600,
            fontFamily: T.sans,
            cursor: loading ? "wait" : "pointer",
            opacity: loading ? 0.7 : 1,
            transition: "opacity 0.15s",
          }}
        >
          {loading ? "Generating..." : "Generate llm-d Manifests"}
        </button>

        {error && (
          <div
            style={{
              padding: "10px 14px",
              borderRadius: 6,
              background: "rgba(239,68,68,0.1)",
              border: `1px solid ${T.accent}44`,
              color: T.accent,
              fontSize: 12,
            }}
          >
            {error}
          </div>
        )}
      </div>

      {/* YAML output */}
      {manifest && (
        <div style={{ borderTop: `1px solid ${T.border}` }}>
          <div
            style={{
              padding: "10px 16px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              borderBottom: `1px solid ${T.border}`,
            }}
          >
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: T.text, fontFamily: T.sans }}>
                Generated Manifests
              </span>
              <span
                style={{
                  fontSize: 10,
                  color: T.textMuted,
                  fontFamily: T.mono,
                  background: T.surface,
                  padding: "2px 8px",
                  borderRadius: 4,
                }}
              >
                {manifest.resource_count} resources
              </span>
            </div>
            <button
              onClick={handleCopy}
              style={{
                padding: "5px 12px",
                borderRadius: 4,
                border: `1px solid ${copied ? T.green + "66" : T.border}`,
                background: copied ? "rgba(34,197,94,0.1)" : T.surface,
                color: copied ? T.green : T.text,
                fontSize: 11,
                fontFamily: T.mono,
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {copied ? "Copied" : "Copy YAML"}
            </button>
          </div>
          <pre
            style={{
              padding: 16,
              margin: 0,
              fontFamily: T.mono,
              fontSize: 11,
              lineHeight: 1.6,
              color: T.teal,
              background: "#0a0c10",
              overflowX: "auto",
              maxHeight: 480,
              overflowY: "auto",
            }}
          >
            {manifest.yaml}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── Main App ──

export default function App() {
  const [systems, setSystems] = useState<SystemInfo[]>([]);
  const [choices, setChoices] = useState<Record<string, string | null>>({
    model: null,
    gpus: null,
    system: null,
    workload: null,
    objective: null,
  });
  const [step, setStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecommendResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchSystems()
      .then(setSystems)
      .catch(() =>
        setSystems([
          { id: "h100_sxm", label: "NVIDIA H100 SXM" },
          { id: "h200_sxm", label: "NVIDIA H200 SXM" },
          { id: "a100_sxm", label: "NVIDIA A100 SXM" },
        ])
      );
  }, []);

  const systemOptions: TreeOption[] = systems.map((s) => ({
    label: s.label,
    value: s.id,
  }));

  const treeSteps: TreeNode[] = [
    {
      id: "model",
      question: "Which model are you deploying?",
      hint: "Determines parallelism strategy and memory requirements",
      options: MODELS,
    },
    {
      id: "gpus",
      question: "How many GPUs are available?",
      hint: "Total GPU count across all nodes",
      options: GPU_COUNTS,
    },
    {
      id: "system",
      question: "What GPU hardware?",
      hint: "Performance data is hardware-specific",
      options: systemOptions,
    },
    {
      id: "workload",
      question: "What workload shape?",
      hint: "Input/output sequence lengths drive the optimal config",
      options: WORKLOADS,
    },
    {
      id: "objective",
      question: "Optimization target?",
      hint: "Throughput and latency rarely optimize together",
      options: OBJECTIVES,
    },
  ];

  const stepNames = ["Model", "GPUs", "System", "Workload", "Target"];

  function handleSelect(nodeId: string, value: string) {
    const newChoices = { ...choices, [nodeId]: value };
    const idx = treeSteps.findIndex((s) => s.id === nodeId);
    treeSteps.slice(idx + 1).forEach((s) => (newChoices[s.id] = null));
    setChoices(newChoices);
    setResult(null);
    setError(null);

    if (idx < treeSteps.length - 1) {
      setStep(idx + 1);
    } else {
      runRecommend(newChoices);
    }
  }

  async function runRecommend(c: Record<string, string | null>) {
    setLoading(true);
    setError(null);
    setStep(treeSteps.length);

    const wl = WORKLOAD_PARAMS[c.workload ?? "long-prefill"];
    const obj = OBJECTIVE_PARAMS[c.objective ?? "balanced"];

    try {
      const res = await recommend({
        model: c.model!,
        gpus: parseInt(c.gpus!, 10),
        system: c.system!,
        isl: wl.isl,
        osl: wl.osl,
        ttft: obj.ttft,
        tpot: obj.tpot,
        prefix: 0,
        top_n: 5,
      });
      setResult(res);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth" }), 200);
    } catch (e: any) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setChoices({ model: null, gpus: null, system: null, workload: null, objective: null });
    setStep(0);
    setResult(null);
    setError(null);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: T.bg,
        color: T.text,
        fontFamily: T.sans,
      }}
    >
      <style>{`
        @keyframes blink { 0%,50%{opacity:1} 51%,100%{opacity:0} }
        @keyframes spin { to { transform: rotate(360deg) } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width:6px; height:6px }
        ::-webkit-scrollbar-track { background: ${T.bg} }
        ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius:3px }
      `}</style>

      {/* Header */}
      <header
        style={{
          padding: "16px 28px",
          borderBottom: `1px solid ${T.border}`,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: T.surface,
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: T.accent,
                boxShadow: `0 0 8px ${T.accent}88`,
              }}
            />
            <h1
              style={{
                fontSize: 20,
                fontWeight: 800,
                color: T.white,
                letterSpacing: -0.5,
              }}
            >
              IntelliConfig
            </h1>
            <span
              style={{
                background: T.accentSoft,
                color: T.accent,
                padding: "2px 10px",
                borderRadius: 4,
                fontSize: 10,
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: 1,
              }}
            >
              MVP
            </span>
          </div>
          <p style={{ color: T.textMuted, fontSize: 12, marginTop: 2 }}>
            vLLM serving optimizer for llm-d · powered by aiconfigurator
          </p>
        </div>
        <button
          onClick={handleReset}
          style={{
            background: T.card,
            border: `1px solid ${T.border}`,
            borderRadius: 6,
            padding: "7px 14px",
            color: T.text,
            fontSize: 12,
            cursor: "pointer",
            fontFamily: T.mono,
          }}
        >
          ↺ Reset
        </button>
      </header>

      {/* Main grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          minHeight: "calc(100vh - 64px)",
        }}
      >
        {/* Left: Decision tree */}
        <div
          style={{
            padding: "24px 28px",
            borderRight: `1px solid ${T.border}`,
            overflowY: "auto",
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: T.textMuted,
              textTransform: "uppercase",
              letterSpacing: 1.5,
              fontFamily: T.mono,
              marginBottom: 16,
            }}
          >
            Configuration Navigator
          </div>

          <StepIndicator steps={stepNames} current={step} />

          {treeSteps.map((node, i) => {
            if (i > step && !choices[node.id]) return null;
            return (
              <TreeStep
                key={node.id}
                node={node}
                selected={choices[node.id]}
                onSelect={(val) => handleSelect(node.id, val)}
                active={i <= step}
              />
            );
          })}

          {loading && (
            <div
              style={{
                marginTop: 20,
                padding: 20,
                textAlign: "center",
                color: T.teal,
              }}
            >
              <div
                style={{
                  width: 24,
                  height: 24,
                  border: `3px solid ${T.border}`,
                  borderTop: `3px solid ${T.teal}`,
                  borderRadius: "50%",
                  animation: "spin 0.8s linear infinite",
                  margin: "0 auto 12px",
                }}
              />
              Searching configuration space...
            </div>
          )}

          {error && (
            <div
              style={{
                marginTop: 20,
                padding: "14px 16px",
                background: "rgba(239,68,68,0.1)",
                border: `1px solid ${T.accent}44`,
                borderRadius: 8,
                color: T.accent,
                fontSize: 13,
              }}
            >
              Error: {error}
            </div>
          )}

          {result && (
            <div
              style={{
                marginTop: 20,
                padding: "14px 16px",
                background: T.tealSoft,
                border: `1px solid ${T.teal}44`,
                borderRadius: 8,
              }}
            >
              <div style={{ color: T.teal, fontWeight: 600, fontSize: 13 }}>
                Recommendation ready
              </div>
              <div style={{ color: T.textMuted, fontSize: 11, marginTop: 2 }}>
                Best mode: <strong style={{ color: T.white }}>{result.chosen_mode}</strong>
                {result.speedup && ` (${result.speedup.toFixed(2)}x speedup)`}
              </div>
            </div>
          )}
        </div>

        {/* Right: CLI preview + results */}
        <div style={{ padding: "24px 28px", overflowY: "auto", background: `${T.bg}e0` }}>
          <div
            style={{
              fontSize: 10,
              color: T.textMuted,
              textTransform: "uppercase",
              letterSpacing: 1.5,
              fontFamily: T.mono,
              marginBottom: 16,
            }}
          >
            CLI Output · Live Preview
          </div>

          <CLIPreview choices={choices} />

          {result && (
            <div
              ref={resultRef}
              style={{ marginTop: 20, display: "flex", flexDirection: "column", gap: 16 }}
            >
              {/* Metrics */}
              <div style={{ display: "flex", gap: 10 }}>
                <MetricCard
                  label="TTFT p99"
                  value={`${result.ttft_ms.toFixed(0)}ms`}
                  color={T.teal}
                />
                <MetricCard
                  label="Throughput"
                  value={`${result.best_throughput_per_gpu.toFixed(0)} tok/s/gpu`}
                  color={T.green}
                />
                <MetricCard
                  label="vs Alternative"
                  value={result.speedup ? `${result.speedup.toFixed(2)}x` : "N/A"}
                  color={T.accent}
                />
              </div>

              <ConfigTable title="Aggregated Configs" configs={result.agg_configs} />
              <ConfigTable title="Disaggregated Configs" configs={result.disagg_configs} />

              {/* llm-d deploy panel */}
              <DeployPanel result={result} />
            </div>
          )}

          {!result && !loading && (
            <div
              style={{
                marginTop: 60,
                textAlign: "center",
                padding: 40,
                color: T.textDim,
                fontSize: 13,
              }}
            >
              <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>⚡</div>
              Navigate the decision tree to generate your recommendation
              <div style={{ fontSize: 11, marginTop: 6, color: T.textDim }}>
                Each choice builds the CLI command — backed by real performance modeling
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
