import Card from "../components/card"
import ChatPage from "../components/chat_page"
import ModelPage from "../components/model_page"
import ToolPage from "../components/tool_page"
import WorkflowPage from "../components/workflow_page"
import Monitor from "../components/monitor"
import LeftNav from "../components/left_nav"
import { useState } from "react"

export default function Dashboard(){
    const [page, setPage] = useState<"chat" | "models" | "tools" | "workflows">("chat");


    return (
        <div>
            <LeftNav page={page} onNavigate={setPage}/>
            {page === "chat" ? (
                <ChatPage/>
            ) : page === "models" ? (
                <ModelPage/>
            ) : page === "tools" ? (
                <ToolPage/>
            ) : page === "workflows" ? (
                <WorkflowPage/>
            ) : null}
            <Monitor/>
        </div>
    )
}
