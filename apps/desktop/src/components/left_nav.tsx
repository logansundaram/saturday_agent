type Page =
    | "chat"
    | "models"
    | "tools"
    | "builder"
    | "workflows"
    | "local_docs"
    | "inspect";

interface LeftNavProps{
    page : Page;
    onNavigate: (page: Page) => void;
}

export default function LeftNav({page, onNavigate} : LeftNavProps){
    const itemClass = (item: Page) =>
        "rounded-lg p-2 transition " +
        (page === item ? "bg-[#6d28d9]/20 text-primary" : "hover:bg-[#6d28d9]/15");

    return (
        <div className="fixed top-0 left-0 h-full bg-root w-50 p-4 z-50">
            <div className="text-left gap-y-2 grid">
                <p className={itemClass("chat")} onClick={() => onNavigate("chat")}>
                    Workspace
                </p>
                <p className={itemClass("models")} onClick={() => onNavigate("models")}>
                    Models
                </p>
                <p className={itemClass("tools")} onClick={() => onNavigate("tools")}>
                    Tools
                </p>
                <p className={itemClass("builder")} onClick={() => onNavigate("builder")}>
                    Builder
                </p>
                <p className={itemClass("workflows")} onClick={() => onNavigate("workflows")}>
                    Workflows
                </p>
                <p className={itemClass("local_docs")} onClick={() => onNavigate("local_docs")}>
                    Local Docs
                </p>
                <p className={itemClass("inspect")} onClick={() => onNavigate("inspect")}>
                    Inspect
                </p>
            </div>
        </div>
    )
}
