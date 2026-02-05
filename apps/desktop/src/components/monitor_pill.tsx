type Monitor_PillProps = {
  header: string
  name: string
  usage: number
}

export default function Monitor_Pill({ header, name, usage }: Monitor_PillProps) {
  const barColor =
    usage >= 100
      ? "bg-red-600"
      : usage >= 90
      ? "bg-orange-500"
      : usage < 80
      ? "bg-green-500"
      : "bg-yellow-400"

  return (
    <div className="m-2 p-4 grid gap-3 rounded-lg border text-left">
      <div className="text-sm font-medium text-slate-700">
        {header} · {name}
      </div>

      <div className="h-2 w-full rounded bg-slate-200 overflow-hidden">
        <span
          className={`block h-full transition-all ${barColor}`}
          style={{ width: `${Math.min(usage, 100)}%` }}
        />
      </div>
    </div>
  )
}
