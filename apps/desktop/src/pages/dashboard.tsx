import { useState } from "react";
import ChatPage from "./ChatPage";
import InspectPage from "./InspectPage";
import ModelsPage from "./ModelsPage";
import ToolPage from "../components/tool_page";
import WorkflowPage from "../components/workflow_page";
import Monitor from "../components/monitor";
import LeftNav from "../components/left_nav";

type Page = "chat" | "models" | "tools" | "workflows" | "inspect";

export default function Dashboard() {
  const [page, setPage] = useState<Page>("chat");
  const [selectedInspectRunId, setSelectedInspectRunId] = useState<string | null>(
    null
  );

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
      ) : page === "workflows" ? (
        <WorkflowPage />
      ) : page === "inspect" ? (
        <InspectPage runId={selectedInspectRunId} onBack={() => setPage("chat")} />
      ) : null}
      <Monitor />
    </div>
  );
}
