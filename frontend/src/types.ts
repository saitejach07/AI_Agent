export type ChatTurn = {
  role: "user" | "assistant";
  content: string;
};

export type AgentTraceStep = {
  step: number;
  tool: string;
  input: Record<string, unknown>;
  output: unknown;
};

export type ChatResponse = {
  answer: string;
  trace: AgentTraceStep[] | null;
};

export type DocumentInfo = {
  source_id: string;
  filename: string;
  chunks: number;
};
