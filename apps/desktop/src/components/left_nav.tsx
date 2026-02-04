import Chats from "./chats"

export default function LeftNav(){
    return (
        <div className="fixed top-0 left-0 h-full bg-slate-200 w-50 p-4">
            <div className="text-left gap-y-2 grid">
                <p>
                    New Chat
                </p>
                <p>
                    Models
                </p>
                <p>
                    Tools
                </p>
                <p>
                    Workflows
                </p>
            </div>
            <Chats/>
        </div>
    )
}