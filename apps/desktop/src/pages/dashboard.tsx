import Card from "../components/card"
import ChatPage from "../components/chat_page"
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
                <Card header="model" body="this is a test card"/>
            ) : page === "tools" ? (
                <Card header="tools" body="this is a test card"/>
            ) : page === "workflows" ? (
                <Card header="workflows" body="this is a test card"/>
            ) : null}
            <Monitor/>
        </div>
    )
}
