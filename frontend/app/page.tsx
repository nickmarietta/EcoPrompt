"use client";

import { useMemo, useState } from "react";

type TaskType = "Explain" | "Summarize" | "Analyze" | "Generate";
type TargetModel = "GPT-4" | "Claude" | "LLaMA";

type OptimizationStats = {
  beforeTokens: number;
  afterTokens: number;
  reductionPct: number;
};

function estimateTokens(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  const words = trimmed.split(/\s+/).filter(Boolean).length;
  return Math.ceil(words * 1.3);
}

/**
 * Frontend-only compression: strip common filler, normalize whitespace.
 */
function optimizePrompt(raw: string): string {
  let out = raw;

  const replacements: RegExp[] = [
    /\bcan you please\b/gi,
    /\bI want you to\b/gi,
    /\bin detail\b/gi,
    /\bplease\b/gi,
  ];

  for (const pattern of replacements) {
    out = out.replace(pattern, " ");
  }

  return out.replace(/\s+/g, " ").trim();
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("Explain");
  const [targetModel, setTargetModel] = useState<TargetModel>("GPT-4");
  const [optimized, setOptimized] = useState("");
  const [stats, setStats] = useState<OptimizationStats | null>(null);

  const hasOutput = optimized.trim().length > 0;

  const efficiencyLabel = useMemo(() => {
    if (!stats) return null;
    return stats.reductionPct > 50 ? "HIGH" : "MEDIUM";
  }, [stats]);

  const handleOptimize = () => {
    const cleaned = optimizePrompt(prompt);
    const beforeTokens = estimateTokens(prompt);
    const afterTokens = estimateTokens(cleaned);

    const reductionPct =
      beforeTokens > 0
        ? Math.round(((beforeTokens - afterTokens) / beforeTokens) * 100)
        : 0;

    setOptimized(cleaned);
    setStats({
      beforeTokens,
      afterTokens,
      reductionPct,
    });
  };

  const copyToClipboard = async () => {
    if (!hasOutput) return;
    await navigator.clipboard.writeText(optimized);
  };

  return (
    <div className="min-h-screen bg-[#040d1b] text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-50">
      <div className="mx-auto flex min-h-screen max-w-[1600px] flex-col gap-6 p-6 lg:flex-row lg:gap-0 lg:p-10">
        {/* LEFT — Input */}
        <section className="flex w-full flex-col lg:w-1/2 lg:border-r lg:border-white/[0.08] lg:pr-10">
          <div className="rounded-2xl border border-white/10 bg-white/[0.06] p-6 shadow-[0_8px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl backdrop-saturate-150 transition hover:border-cyan-400/25 hover:bg-white/[0.08]">
            <header className="mb-8">
              <h1 className="text-balance font-semibold tracking-tight text-white">
                <span className="text-3xl sm:text-4xl">🌿 EcoPrompt</span>
              </h1>
              <p className="mt-3 max-w-xl text-sm leading-relaxed text-cyan-100/75">
                Shrink prompts without losing intent—fewer tokens, clearer
                instructions, and lower AI compute cost on{" "}
                <span className="text-cyan-300/90">{targetModel}</span> for{" "}
                <span className="text-cyan-300/90">{taskType}</span> tasks.
              </p>
            </header>

            <div className="space-y-5">
              <label className="block">
                <span className="mb-2 block text-xs font-medium uppercase tracking-wider text-cyan-300/70">
                  Your prompt
                </span>
                <textarea
                  className="min-h-[140px] w-full resize-y rounded-xl border border-white/10 bg-white/[0.07] px-4 py-3 text-[15px] leading-relaxed text-slate-100 outline-none ring-cyan-400/40 transition placeholder:text-slate-500 focus:border-cyan-400/35 focus:ring-2"
                  placeholder="Paste a verbose prompt—we’ll return a tighter version with the same intent..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  spellCheck
                />
              </label>

              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block">
                  <span className="mb-2 block text-xs font-medium uppercase tracking-wider text-cyan-300/70">
                    Task type
                  </span>
                  <select
                    className="w-full cursor-pointer rounded-xl border border-white/10 bg-white/[0.07] px-4 py-3 text-sm text-slate-100 outline-none transition hover:bg-white/[0.09] focus:border-cyan-400/35 focus:ring-2 focus:ring-cyan-400/40"
                    value={taskType}
                    onChange={(e) => setTaskType(e.target.value as TaskType)}
                  >
                    <option value="Explain">Explain</option>
                    <option value="Summarize">Summarize</option>
                    <option value="Analyze">Analyze</option>
                    <option value="Generate">Generate</option>
                  </select>
                </label>

                <label className="block">
                  <span className="mb-2 block text-xs font-medium uppercase tracking-wider text-cyan-300/70">
                    Target model
                  </span>
                  <select
                    className="w-full cursor-pointer rounded-xl border border-white/10 bg-white/[0.07] px-4 py-3 text-sm text-slate-100 outline-none transition hover:bg-white/[0.09] focus:border-cyan-400/35 focus:ring-2 focus:ring-cyan-400/40"
                    value={targetModel}
                    onChange={(e) =>
                      setTargetModel(e.target.value as TargetModel)
                    }
                  >
                    <option value="GPT-4">GPT-4</option>
                    <option value="Claude">Claude</option>
                    <option value="LLaMA">LLaMA</option>
                  </select>
                </label>
              </div>

              <button
                type="button"
                onClick={handleOptimize}
                className="w-full rounded-xl bg-gradient-to-r from-cyan-400 to-cyan-300 px-5 py-3.5 text-sm font-semibold text-[#031018] shadow-[0_0_24px_rgba(34,211,238,0.25)] transition hover:from-cyan-300 hover:to-cyan-200 hover:shadow-[0_0_32px_rgba(34,211,238,0.35)] active:scale-[0.99]"
              >
                Optimize Prompt
              </button>
            </div>
          </div>
        </section>

        {/* RIGHT — Output */}
        <section className="flex w-full flex-col lg:w-1/2 lg:pl-10">
          <div className="flex min-h-full flex-col rounded-2xl border border-white/10 bg-white/[0.06] p-6 shadow-[0_8px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl backdrop-saturate-150 transition hover:border-cyan-400/25 hover:bg-white/[0.08]">
            <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="text-xl font-semibold tracking-tight text-white">
                  Optimized Output
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Same intent—compressed for fewer tokens.
                </p>
              </div>
              <button
                type="button"
                disabled={!hasOutput}
                onClick={copyToClipboard}
                className="rounded-lg border border-white/15 bg-white/[0.06] px-4 py-2 text-sm font-medium text-cyan-100 transition hover:border-cyan-400/40 hover:bg-white/[0.1] disabled:cursor-not-allowed disabled:opacity-40"
              >
                Copy
              </button>
            </header>

            <div className="mb-6 min-h-[140px] rounded-xl border border-white/[0.08] bg-black/25 px-4 py-3 font-mono text-[13px] leading-relaxed text-cyan-50/95">
              {hasOutput ? (
                optimized
              ) : (
                <span className="text-slate-500">
                  Run optimize to see your tightened prompt here.
                </span>
              )}
            </div>

            {stats && (
              <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-3">
                <div className="rounded-xl border border-white/[0.07] bg-white/[0.05] px-4 py-3 transition hover:border-cyan-400/25">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-cyan-300/65">
                    Tokens
                  </p>
                  <p className="mt-1 font-mono text-lg font-semibold tabular-nums text-white">
                    {stats.beforeTokens}{" "}
                    <span className="text-cyan-400/80">→</span>{" "}
                    {stats.afterTokens}
                  </p>
                </div>
                <div className="rounded-xl border border-white/[0.07] bg-white/[0.05] px-4 py-3 transition hover:border-cyan-400/25">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-cyan-300/65">
                    Reduction
                  </p>
                  <p className="mt-1 font-mono text-lg font-semibold tabular-nums text-white">
                    {stats.reductionPct}%
                  </p>
                </div>
                <div className="rounded-xl border border-white/[0.07] bg-white/[0.05] px-4 py-3 transition hover:border-cyan-400/25">
                  <p className="text-[11px] font-medium uppercase tracking-wider text-cyan-300/65">
                    Efficiency
                  </p>
                  <p className="mt-1 text-lg font-semibold text-cyan-300">
                    {efficiencyLabel}
                  </p>
                </div>
              </div>
            )}

            {stats && (
              <div className="mb-6 rounded-xl border border-cyan-400/20 bg-gradient-to-br from-cyan-500/10 to-transparent px-4 py-4">
                <h3 className="text-sm font-semibold text-cyan-200">
                  🧬 Human Delta
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-cyan-100/80">
                  Fewer tokens mean less model work per request. This prompt
                  trims roughly{" "}
                  <span className="font-semibold text-cyan-300">
                    {stats.reductionPct}%
                  </span>{" "}
                  of estimated token overhead—same goal, lower AI compute.
                </p>
              </div>
            )}

            <div className="mt-auto border-t border-white/[0.07] pt-6">
              <h3 className="text-sm font-semibold text-white">What Changed</h3>
              <ul className="mt-3 space-y-2 text-sm text-slate-300">
                <li className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-cyan-400/80" />
                  Removed filler words
                </li>
                <li className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-cyan-400/80" />
                  Reduced redundancy
                </li>
                <li className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-cyan-400/80" />
                  Improved clarity
                </li>
                <li className="flex gap-2">
                  <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-cyan-400/80" />
                  Compressed structure
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
