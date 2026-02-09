import type { ChatTransport } from "./chatTransport";
import { LangGraphApiTransport } from "./transports/langgraphApiTransport";
import { OllamaTransport } from "./transports/ollamaTransport";

export function getChatTransport(): ChatTransport {
  const selectedTransport = (import.meta.env.VITE_CHAT_TRANSPORT ?? "langgraph")
    .toLowerCase()
    .trim();

  if (selectedTransport === "ollama") {
    return new OllamaTransport();
  }

  return new LangGraphApiTransport();
}
