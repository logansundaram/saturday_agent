import Header from "./header"

export default function ToolCard() {
  return (
    <div className="w-60 h-40 rounded-xl border border-subtle bg-panel p-4 flex flex-col">
      <div className="space-y-1">
        <div className="text-sm font-medium tracking-tight">
          File System
        </div>
        <div className="text-xs text-muted-foreground">
          Local Tool
        </div>
      </div>

      <div className="mt-4 text-xs text-muted-foreground leading-relaxed">
        Read, write, and manage files on the local machine.
      </div>

      <div className="mt-auto pt-3 border-t border-subtle flex justify-between items-center text-xs">
        <span className="text-muted-foreground">
          Status
        </span>
        <span className="font-medium text-gold">
          Enabled
        </span>
      </div>
    </div>
  )
}
