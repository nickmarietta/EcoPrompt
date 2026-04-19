"use client";

/**
 * Short honest pitch + optional detail disclosure (not literal per-prompt impact).
 */
export default function EcoPromptExplainer() {
  return (
    <aside
      className="text-sm leading-relaxed"
      aria-labelledby="ecoprompt-water-heading"
    >
      <h3
        id="ecoprompt-water-heading"
        className="text-xs font-semibold uppercase tracking-[0.12em] text-sky-200/90 sm:text-sm"
      >
        How this helps save water
      </h3>
      <p className="mt-2 text-slate-300 sm:text-[15px] sm:leading-relaxed">
        AI runs on power-hungry infrastructure—including cooling that can be
        water-intensive.{" "}
        <span className="font-semibold text-slate-100">EcoPrompt</span> tightens
        prompts so you burn fewer tokens and fewer retries; we show water and energy
        as simple illustrative estimates, not a meter reading for one prompt.
      </p>

      <details className="mt-2 group">
        <summary className="cursor-pointer list-none text-sm font-medium text-sky-400 outline-none hover:text-sky-300 [&::-webkit-details-marker]:hidden">
          <span className="inline-flex items-center gap-1.5">
            <span className="transition group-open:rotate-90" aria-hidden>
              ▸
            </span>
            Learn more
          </span>
        </summary>
        <div className="mt-3 space-y-3 border-l border-white/10 pl-4 text-sm text-slate-400">
          <p>
            Clearer, more efficient prompts can mean less wasted compute. Less waste
            can, over many runs, nudge down the energy and cooling burden behind model
            hosting—including indirect pressure on the water systems data centers rely
            on. None of that is the same as “saving the ocean” from a single rewrite;
            it’s a small link in a long chain.
          </p>
          <p>
            <span className="font-semibold text-slate-300">Note:</span> Values here are
            estimated proxies from prompt efficiency, not direct measurements of
            real-world water use for this request alone.
          </p>
          <p>
            The goal is a slightly more sustainable digital habit: less unnecessary AI
            compute, and a bit less load on the shared energy and water systems that
            ultimately touch the wider environment.
          </p>
        </div>
      </details>
    </aside>
  );
}
