import { clusterRequest } from '@kinvolk/headlamp-plugin/lib/ApiProxy';
import type { PluginConfig } from '../utils';

/**
 * Default base URL for the Holmes server (direct / port-forward fallback).
 */
export const DEFAULT_AGUI_URL = 'http://localhost:5050';

/**
 * Holmes Kubernetes Service details.
 * Must match the Service resource deployed by the Holmes Helm chart.
 */
export const HOLMES_SERVICE_NAME = 'holmesgpt-holmes';
export const HOLMES_SERVICE_PORT = 80;
export const HOLMES_SERVICE_NAMESPACE = 'default';

function normalizeConfigString(value: string | undefined, fallback: string): string {
  const normalized = value?.trim();
  return normalized || fallback;
}

function normalizeConfigPort(value: number | undefined, fallback: number): number {
  return Number.isInteger(value) && value >= 1 && value <= 65535 ? value : fallback;
}

function getHolmesServiceConfig(config?: PluginConfig): {
  namespace: string;
  serviceName: string;
  servicePort: number;
} {
  return {
    namespace: normalizeConfigString(config?.holmesNamespace, HOLMES_SERVICE_NAMESPACE),
    serviceName: normalizeConfigString(config?.holmesServiceName, HOLMES_SERVICE_NAME),
    servicePort: normalizeConfigPort(config?.holmesPort, HOLMES_SERVICE_PORT),
  };
}

/**
 * Build the K8s API path that proxies to the Holmes service.
 *
 * Path pattern:
 *   /api/v1/namespaces/{ns}/services/{svc}:{port}/proxy[/{subPath}]
 */
export function getHolmesServiceProxyPath(config?: PluginConfig, subPath = ''): string {
  const { namespace, serviceName, servicePort } = getHolmesServiceConfig(config);
  const base = `/api/v1/namespaces/${namespace}/services/${serviceName}:${servicePort}/proxy`;
  return subPath ? `${base}/${subPath.replace(/^\//, '')}` : base;
}

/**
 * Check if the Holmes agent is reachable via the K8s service proxy.
 */
export async function checkHolmesAgentHealth(
  cluster: string,
  config?: PluginConfig
): Promise<boolean> {
  try {
    await clusterRequest(getHolmesServiceProxyPath(config, ''), {
      cluster,
      isJSON: false,
      timeout: 5000,
    });
    return true;
  } catch (err: any) {
    const status = err?.status;
    if (status === 404 || status === 405 || status === 422) {
      return true;
    }
    return false;
  }
}

/**
 * Resolve the Headlamp backend origin.
 */
function getHeadlampBackendOrigin(): string {
  if (
    typeof window !== 'undefined' &&
    ((typeof window.process === 'object' && (window.process as any).type === 'renderer') ||
      (typeof navigator === 'object' && navigator.userAgent.indexOf('Electron') >= 0))
  ) {
    const port = (window as any).headlampBackendPort || 4466;
    return `http://localhost:${port}`;
  }

  if (typeof window !== 'undefined' && (window as any).ddClient !== undefined) {
    return 'http://localhost:64446';
  }

  try {
    if ((import.meta as any).env?.DEV) {
      return 'http://localhost:4466';
    }
  } catch {
    // import.meta may not be available in all contexts
  }

  return window.location.origin;
}

/**
 * Build the full Holmes base URL that routes through Headlamp's backend
 * proxy → Kubernetes API server → Holmes Service.
 */
export function getHolmesProxyBaseUrl(cluster: string, config?: PluginConfig): string {
  const origin = getHeadlampBackendOrigin();
  let baseUrlPrefix = '';
  if (typeof window !== 'undefined' && (window as any).headlampBaseUrl) {
    const raw = (window as any).headlampBaseUrl as string;
    if (raw !== '/' && raw !== './' && raw !== '.') {
      baseUrlPrefix = raw;
    }
  }
  return `${origin}${baseUrlPrefix}/clusters/${cluster}${getHolmesServiceProxyPath(config, '')}`;
}

// ─── Holmes SSE event shapes ───────────────────────────────────────────────

interface HolmesStartToolCallingEvent {
  tool_name: string;
  id: string;
}

interface HolmesToolCallingResultEvent {
  tool_call_id: string;
  tool_name: string;
}

interface HolmesAiAnswerEndEvent {
  analysis: string;
  conversation_history: object[];
}


// ─── Subscriber interface ──────────────────────────────────────────────────

interface HolmesSubscriber {
  onEvent?: (args: { event: any }) => void;
  onRunInitialized?: () => void;
  onRunFailed?: (args: { error: any }) => void;
  onRunFinalized?: () => void;
  onRunStartedEvent?: () => void;
  onRunFinishedEvent?: () => void;
  onRunErrorEvent?: (args: { event: { message: string } }) => void;
  onTextMessageStartEvent?: (args: { event: { messageId: string } }) => void;
  onTextMessageContentEvent?: (args: { event: { delta: string } }) => void;
  onTextMessageEndEvent?: () => void;
  onToolCallStartEvent?: (args: { event: { toolCallName: string; toolCallId?: string } }) => void;
  onToolCallEndEvent?: (args: { toolCallName: string }) => void;
}

/**
 * HolmesAgent talks to the Holmes /api/chat SSE endpoint.
 *
 * Mapping from Holmes SSE events to subscriber callbacks:
 *   start_tool_calling   → onToolCallStartEvent
 *   tool_calling_result  → onToolCallEndEvent
 *   ai_answer_end        → onTextMessage{Start,Content,End}Event + onRunFinishedEvent
 *   token_count          → ignored
 *   ai_message           → ignored (channel-marker format; clean text comes from ai_answer_end)
 *
 * Conversation history is preserved across calls via ai_answer_end.conversation_history,
 * enabling multi-turn chat. Call resetThread() to start a fresh conversation.
 */
export class HolmesAgent {
  private baseUrl: string;
  private threadId: string;
  private subscribers: HolmesSubscriber[] = [];
  private conversationHistory: object[] = [];
  private pendingAsk: string = '';
  private abortController: AbortController | null = null;

  constructor(baseUrl: string = DEFAULT_AGUI_URL) {
    this.baseUrl = baseUrl;
    this.threadId = `thread-${Date.now()}`;
  }

  get connectionLabel(): string {
    return this.baseUrl;
  }

  subscribe(subscriber: HolmesSubscriber): { unsubscribe: () => void } {
    this.subscribers.push(subscriber);
    return {
      unsubscribe: () => {
        this.subscribers = this.subscribers.filter(s => s !== subscriber);
      },
    };
  }

  addMessage(message: { id: string; role: string; content: string }): void {
    if (message.role === 'user') {
      this.pendingAsk = message.content;
    }
  }

  async runAgent(params?: { runId?: string; tools?: any[]; context?: any[]; forwardedProps?: Record<string, any> }): Promise<void> {
    const ask = this.pendingAsk;
    this.pendingAsk = '';

    if (!ask) {
      this.emit('onRunErrorEvent', { event: { message: 'No message to send' } });
      return;
    }

    const url = `${this.baseUrl}/api/chat`;
    const body: Record<string, any> = { ask, stream: true };
    if (this.conversationHistory.length > 0) {
      body.conversation_history = this.conversationHistory;
    }

    this.abortController = new AbortController();
    this.emit('onRunStartedEvent');

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: this.abortController.signal,
      });

      if (!response.ok) {
        const text = await response.text().catch(() => response.statusText);
        throw new Error(`HTTP ${response.status}: ${text}`);
      }

      if (!response.body) {
        throw new Error('Response has no body');
      }

      await this.parseSSEStream(response.body);
      this.emit('onRunFinishedEvent');
    } catch (err: any) {
      if (err.name === 'AbortError') {
        this.emit('onRunFinishedEvent');
      } else {
        this.emit('onRunErrorEvent', { event: { message: err?.message ?? 'Unknown error' } });
      }
    } finally {
      this.abortController = null;
    }
  }

  private async parseSSEStream(body: ReadableStream<Uint8Array>): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('event:')) {
            currentEvent = line.slice(6).trim();
          } else if (line.startsWith('data:')) {
            const raw = line.slice(5).trim();
            if (raw && currentEvent) {
              try {
                this.handleHolmesEvent(currentEvent, JSON.parse(raw));
              } catch {
                // skip malformed JSON
              }
            }
            currentEvent = '';
          } else if (line === '') {
            currentEvent = '';
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private handleHolmesEvent(eventType: string, data: any): void {
    switch (eventType) {
      case 'start_tool_calling': {
        const e = data as HolmesStartToolCallingEvent;
        this.emit('onToolCallStartEvent', {
          event: { toolCallName: e.tool_name, toolCallId: e.id },
        });
        break;
      }

      case 'tool_calling_result': {
        const e = data as HolmesToolCallingResultEvent;
        this.emit('onToolCallEndEvent', { toolCallName: e.tool_name });
        break;
      }

      case 'ai_answer_end': {
        const e = data as HolmesAiAnswerEndEvent;
        if (e.conversation_history) {
          this.conversationHistory = e.conversation_history;
        }
        const analysis = e.analysis?.trim() ?? '';
        if (analysis) {
          const msgId = `msg-${Date.now()}`;
          this.emit('onTextMessageStartEvent', { event: { messageId: msgId } });
          this.emit('onTextMessageContentEvent', { event: { delta: analysis } });
          this.emit('onTextMessageEndEvent');
        }
        break;
      }

      // token_count and ai_message are informational; ignored.
    }
  }

  private emit(eventName: string, ...args: any[]): void {
    for (const sub of this.subscribers) {
      const fn = (sub as any)[eventName];
      if (typeof fn === 'function') {
        fn(...args);
      }
    }
  }

  abortRun(): void {
    this.abortController?.abort();
  }

  resetThread(): void {
    this.threadId = `thread-${Date.now()}`;
    this.conversationHistory = [];
    this.pendingAsk = '';
    this.abortController?.abort();
    this.abortController = null;
  }

  getThreadId(): string {
    return this.threadId;
  }
}
