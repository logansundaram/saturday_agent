export default function WorkflowCard() {
  return (
    <div className="w-60 h-44 rounded-xl border bg-panel px-4 pt-4 pb-3 flex flex-col">
      <div className="space-y-0.5">
        <div className="text-sm font-medium tracking-tight leading-tight">
          Document Q&A
        </div>
        <div className="text-xs text-muted-foreground leading-tight">
          RAG Workflow
        </div>
      </div>

      <div className="mt-3 text-xs text-muted-foreground leading-snug">
        Ingest files, retrieve relevant context, and generate grounded answers.
      </div>

      <div className="mt-3 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">4 steps</span>
        <span className="mx-2 text-muted-foreground">·</span>
        Deterministic
      </div>

      <div className="mt-auto pt-3 border-t flex justify-between items-end text-xs">
        <span className="text-muted-foreground leading-none">
          Status
        </span>
        <span className="font-medium text-green-500 leading-none">
          Ready
        </span>
      </div>
    </div>
  )
}
