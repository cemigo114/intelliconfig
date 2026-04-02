import { useState, useEffect, useRef, type CSSProperties } from "react";
import {
  fetchSystems,
  recommend,
  checkSupport,
  type SystemInfo,
  type RecommendResult,
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

              {/* vLLM serve command */}
              <div
                style={{
                  background: T.card,
                  border: `1px solid ${T.border}`,
                  borderRadius: 10,
                  padding: 16,
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: T.text,
                    marginBottom: 8,
                    fontFamily: T.sans,
                  }}
                >
                  vLLM Serve Command
                </div>
                <pre
                  style={{
                    fontFamily: T.mono,
                    fontSize: 11,
                    color: T.teal,
                    lineHeight: 1.7,
                    background: "#0a0c10",
                    padding: 12,
                    borderRadius: 6,
                    overflowX: "auto",
                  }}
                >
                  {`vllm serve ${result.model_path} \\\n  --tensor-parallel-size ${
                    result.agg_configs[0]?.tp ?? result.disagg_configs[0]?.["(p)tp"] ?? 1
                  }`}
                </pre>
              </div>
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
