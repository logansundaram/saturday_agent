import ToolCard from "./tool_card"


export default function ToolPage(){
    return (
        <div className="bg-panel section-base">
            <div className="section-hero">
                <h1 className="section-header">
                    Tools
                </h1>
                <p className="section-framer">
                    Local tools by default, with optional external integrations. Compose, inspect, and control every action your agent takes.
                </p>
            </div>

            <ToolCard/>

        </div>
    )
}