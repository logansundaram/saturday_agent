import Chats from "./chats"

export default function LeftNav(){
    return (
        <div className="fixed top-0 left-0 h-full bg-slate-200 w-50 p-4">
            <div className="text-left gap-y-2 grid">
                <p className="hover:bg-slate-400 rounded-lg p-2">
                    New Chat
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2">
                    Models
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2">
                    Tools
                </p>
                <p className="hover:bg-slate-400 rounded-lg p-2">
                    Workflows
                </p>
            </div>


            <Chats/>
        </div>
    )
}