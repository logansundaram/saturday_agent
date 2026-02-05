type ModelCardProps = {
    title: string;
    company: string;
    size_param: string;
    size_gb: number;
    strengths: string[];
    source: string;
    rating: string;
}


export default function ModelCard(){
    return(
        <div className="rounded-lg border-1 w-60 h-80">
            <div className="text-xl p-4 text-left">
                <div>
                    LLM Title
                </div>  
                <div className="text-sm">
                    LLM Provider/Company
                </div>
            </div>

            <div className="text-left p-2">
                Model Size: 28GB 
            </div>

            <div className="text-left p-2">
                Strengths: Reasoning, Coding, Agentic
            </div>

            <div className="text-left p-2">
                Source: local
            </div>

            <div className="text-left p-2">
                Rating: Excellent
                <div className="text-sm">
                    This model will run effectively on your machine
                </div>
            </div>

        </div>
    )
}