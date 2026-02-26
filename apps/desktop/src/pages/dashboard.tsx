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

type DashboardNavigateDetail = {
  page?: Page;
  runId?: string;
  sourceRunId?: string;
  origin?: "rerun_from_state" | string;
};

type PendingChatRerun = {
  runId: string;
  sourceRunId?: string;
  origin?: "rerun_from_state" | string;
  nonce: string;
};

function createNonce(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function Dashboard() {
  const [page, setPage] = useState<Page>("chat");
  const [selectedInspectRunId, setSelectedInspectRunId] = useState<string | null>(
    null
  );
  const [pendingChatRerun, setPendingChatRerun] = useState<PendingChatRerun | null>(
    null
  );

  useEffect(() => {
    const handler = (event: Event) => {
      const customEvent = event as CustomEvent<DashboardNavigateDetail>;
      const detail = customEvent.detail ?? {};
      const targetPage = detail.page;
      if (!targetPage) {
        return;
      }
      if (targetPage === "inspect") {
        const nextRunId = String(detail.runId || "").trim();
        if (nextRunId) {
          setSelectedInspectRunId(nextRunId);
        }
        setPendingChatRerun(null);
      } else if (targetPage === "chat") {
        const nextRunId = String(detail.runId || "").trim();
        if (nextRunId) {
          const sourceRunId = String(detail.sourceRunId || "").trim();
          setPendingChatRerun({
            runId: nextRunId,
            sourceRunId: sourceRunId || undefined,
            origin: detail.origin,
            nonce: createNonce(),
          });
        } else {
          setPendingChatRerun(null);
        }
      } else {
        setPendingChatRerun(null);
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
          incomingRerun={pendingChatRerun}
          onIncomingRerunHandled={(nonce) => {
            setPendingChatRerun((current) =>
              current && current.nonce === nonce ? null : current
            );
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
