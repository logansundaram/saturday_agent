type Monitor_PillProps = {
  header: string
  name: string
  usage: number
}

export default function Monitor_Pill({ header, name, usage }: Monitor_PillProps) {
  const barColor =
    usage >= 90
      ? "bg-[#d4af37]"
      : usage >= 70
      ? "bg-[#8b5cf6]"
      : "bg-[#5b21b6]"

  return (
    <div className="m-2 p-4 grid gap-3 rounded-lg border border-subtle text-left">
      <div className="text-sm font-medium text-muted-foreground">
        {header} · {name}
      </div>

      <div className="h-2 w-full rounded bg-[#1b1b1f] overflow-hidden">
        <span
          className={`block h-full transition-all ${barColor}`}
          style={{ width: `${Math.min(usage, 100)}%` }}
        />
      </div>
    </div>
  )
}
