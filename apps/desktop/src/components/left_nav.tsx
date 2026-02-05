import Chats from "./chats"

type Page = "chat" | "models" | "tools" | "workflows";

interface LeftNavProps{
    page : Page;
    onNavigate: (page: Page) => void;
}

export default function LeftNav({page, onNavigate} : LeftNavProps){
    return (
        <div className="fixed top-0 left-0 h-full bg-root w-50 p-4">
            <div className="text-left gap-y-2 grid">
                <p className="hover:bg-slate-400 rounded-lg p-2" onClick={() => onNavigate("chat")}>
                    New Chat
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2" onClick={() => onNavigate("models")}>
                    Models
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2" onClick={() => onNavigate("tools")}>
                    Tools
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2" onClick={() => onNavigate("workflows")}>
                    Workflows
                </p>
            </div>


            <Chats/>
        </div>
    )
}