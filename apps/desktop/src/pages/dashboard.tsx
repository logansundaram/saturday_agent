import { useEffect, useState } from "react";
import ChatPage from "./ChatPage";
import BuilderPage from "./BuilderPage";
import InspectPage from "./InspectPage";
import ModelsPage from "./ModelsPage";
import ToolPage from "../components/tool_page";
import WorkflowPage from "../components/workflow_page";
import Monitor from "../components/monitor";
import LeftNav from "../components/left_nav";
import LocalDocsPage from "./LocalDocsPage";

type Page =
  | "chat"
  | "models"
  | "tools"
  | "builder"
  | "workflows"
  | "local_docs"
  | "inspect";

export default function Dashboard() {
  const [page, setPage] = useState<Page>("chat");
  const [selectedInspectRunId, setSelectedInspectRunId] = useState<string | null>(
    null
  );

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent<{ page?: Page; runId?: string }>;
      const targetPage = customEvent.detail?.page;
      if (!targetPage) {
        return;
      }
      if (targetPage === "inspect") {
        const nextRunId = String(customEvent.detail?.runId || "").trim();
        if (nextRunId) {
          setSelectedInspectRunId(nextRunId);
        }
      }
      setPage(targetPage);
    };

    window.addEventListener("dashboard:navigate", handler);
    return () => {
      window.removeEventListener("dashboard:navigate", handler);
    };
  }, []);

  return (
    <div>
      <LeftNav page={page} onNavigate={setPage} />
      {page === "chat" ? (
        <ChatPage
          onInspectRun={(runId) => {
            setSelectedInspectRunId(runId);
            setPage("inspect");
          }}
        />
      ) : page === "models" ? (
        <ModelsPage />
      ) : page === "tools" ? (
        <ToolPage />
      ) : page === "builder" ? (
        <BuilderPage />
      ) : page === "workflows" ? (
        <WorkflowPage />
      ) : page === "local_docs" ? (
        <LocalDocsPage />
      ) : page === "inspect" ? (
        <InspectPage runId={selectedInspectRunId} onBack={() => setPage("chat")} />
      ) : null}
      <Monitor />
    </div>
  );
}
