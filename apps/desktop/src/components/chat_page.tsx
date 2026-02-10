export default function ChatPage(){
    return (
        <div className="bg-panel section-base relative">
            <div className="section-hero pb-32">
                <h1 className="section-header">
                    Chat
                </h1>
                <p className="section-framer">
                    Ask anything and get a clear, actionable response.
                </p>
            </div>

            <form className="fixed bottom-0 left-[12.5rem] right-0 pb-6">
                <div className="mx-auto max-w-3xl">
                    <div className="flex items-end gap-3 rounded-2xl border border-subtle bg-[#0b0b10]/90 px-4 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.35)] ring-1 ring-white/10 focus-within:ring-2 focus-within:ring-[#6d28d9]/40 transition">
                        <textarea
                            name="message"
                            placeholder="Ask anything"
                            rows={1}
                            className="flex-1 bg-transparent text-sm text-primary placeholder:text-secondary outline-none resize-none min-h-12 max-h-40 leading-relaxed"
                        />
                        <button
                            type="submit"
                            className="h-9 w-9 shrink-0 rounded-full bg-gold text-black hover:bg-[#e1c161] transition"
                            aria-label="Send message"
                        >
                            →
                        </button>
                    </div>
                </div>
            </form>
        </div>
    )
}
