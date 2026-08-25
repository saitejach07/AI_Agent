import type { ChatResponse, ChatTurn, DocumentInfo } from "./types";

const DEFAULT_API_BASE_URL = import.meta.env.PROD
  ? "https://aiagentbe.vercel.app"
  : "http://127.0.0.1:8000";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL).replace(/\/$/, "");

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const response = await fetch(`${API_BASE_URL}/api/documents`);
  return parseResponse<DocumentInfo[]>(response);
}

export async function uploadDocuments(files: FileList): Promise<void> {
  const formData = new FormData();

  Array.from(files).forEach((file) => {
    formData.append("files", file);
  });

  const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
    method: "POST",
    body: formData,
  });

  await parseResponse(response);
}

export async function askQuestion(question: string, history: ChatTurn[]): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agent/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      history,
      max_steps: 8,
      use_web_search: true,
      return_trace: true,
    }),
  });

  return parseResponse<ChatResponse>(response);
}
