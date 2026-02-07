type ModelCardProps = {
    title: string;
    company: string;
    size_param: string;
    size_gb: number;
    strengths: string[];
    source: string;
    rating: string;
}


export default function ModelCard() {
  return (
    <div className="w-80 h-96 rounded-xl border bg-panel p-6 flex flex-col">
      <div className="space-y-1">
        <div className="text-base font-semibold tracking-tight leading-tight">
          Qwen 2.5 32B
        </div>
        <div className="text-sm text-muted-foreground leading-tight">
          Alibaba
        </div>
      </div>

      <div className="mt-6 space-y-5 text-sm">
        <div className="space-y-1">
          <div className="text-muted-foreground">
            Model Size
          </div>
          <div className="font-medium">
            28 GB
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-muted-foreground">
            Strengths
          </div>
          <div className="font-medium">
            Reasoning, Coding, Agentic
          </div>
        </div>

        <div className="space-y-1">
          <div className="text-muted-foreground">
            Source
          </div>
          <div className="font-medium">
            Local
          </div>
        </div>
      </div>

      <div className="mt-auto pt-5 border-t text-sm space-y-1">
        <div className="font-semibold text-green-500">
          Excellent
        </div>
        <div className="text-muted-foreground">
          Runs efficiently on your machine
        </div>
      </div>
    </div>
  )
}


