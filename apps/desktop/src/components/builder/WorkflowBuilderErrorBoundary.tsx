import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "../ui/button";

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  message: string;
};

export default class WorkflowBuilderErrorBoundary extends Component<Props, State> {
  state: State = {
    hasError: false,
    message: "",
  };

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      message: error.message || "Unexpected editor error.",
    };
  }

  componentDidCatch(error: Error, _info: ErrorInfo): void {
    this.setState({
      hasError: true,
      message: error.message || "Unexpected editor error.",
    });
  }

  private handleReset = () => {
    this.setState({ hasError: false, message: "" });
  };

  render(): ReactNode {
    if (!this.state.hasError) {
      return this.props.children;
    }
    return (
      <div className="rounded-xl border border-rose-400/40 bg-rose-500/10 p-4 text-sm text-rose-100">
        <p className="font-medium">Workflow editor crashed.</p>
        <p className="mt-1 text-xs text-rose-100/90">{this.state.message}</p>
        <Button
          type="button"
          className="mt-3 h-8 rounded-full border border-rose-300/40 bg-transparent px-3 text-xs text-rose-100"
          onClick={this.handleReset}
        >
          Retry editor
        </Button>
      </div>
    );
  }
}
