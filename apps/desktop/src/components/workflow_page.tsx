import WorkflowCard from "./workflow_card"


export default function WorkflowPage(){
    return (
        <div className="bg-panel section-base">
            <div className="section-hero">
                <h1 className="section-header">
                    Workflows
                </h1>
                <p className="section-framer">
                    Local, transparent workflows by default. Inspect, replay, and tailor each step to fit your task.
                </p>
            </div>


            <WorkflowCard/>


        </div>
    )
}