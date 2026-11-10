/**
 * Claude Code as a LangChain chat model: run Dexter on your own Claude subscription.
 *
 * No API key. Each model call starts the `claude` command you already have installed
 * and signed in to (`claude -p`), so usage counts against your Claude plan like any
 * other Claude Code use.
 *
 * Why shell out instead of calling the API with your login token: Anthropic's terms
 * allow a Claude plan to be used through the unmodified Claude Code app, and do not
 * allow other software to collect or reuse the plan's OAuth token. This file never
 * reads, stores or forwards a credential. It only starts the `claude` program.
 *
 * The call is deliberately stripped down:
 *  - `--safe-mode` keeps the login and model choice but skips CLAUDE.md files, plugins,
 *    hooks and MCP servers (a one-word reply drops from ~18k input tokens to ~1k).
 *  - `--bare` would be smaller still, but it ignores the subscription login, so it is
 *    never used.
 *  - `--tools ""` turns off Claude Code's own tools. Dexter has its own tools; this
 *    model only decides which of them to call.
 *  - ANTHROPIC_API_KEY & co. are removed from the child's environment, because Claude
 *    Code prefers an API key over the subscription login when one is set.
 *
 * Tool calling is emulated: the bound tools are described in the system prompt and the
 * model answers with JSON (`--json-schema` enforces the shape) that we turn into
 * LangChain `tool_calls`. Structured output works the same way with the caller's schema.
 * Replies arrive whole, not token by token.
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { delimiter, join } from 'node:path';
import {
  BaseChatModel,
  type BaseChatModelCallOptions,
  type BaseChatModelParams,
} from '@langchain/core/language_models/chat_models';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  coerceMessageLikeToMessage,
  type BaseMessage,
} from '@langchain/core/messages';
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs';
import { RunnableLambda, type Runnable } from '@langchain/core/runnables';
import { convertToOpenAITool } from '@langchain/core/utils/function_calling';
import { toJsonSchema } from '@langchain/core/utils/json_schema';
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager';

export const CLAUDE_CODE_MODEL_PREFIX = 'claude-code:';

/** Variables that make Claude Code bill an API account (or another cloud) instead of the subscription. */
export const BILLING_ENV_VARS = [
  'ANTHROPIC_API_KEY',
  'ANTHROPIC_AUTH_TOKEN',
  'ANTHROPIC_BASE_URL',
  'CLAUDE_CODE_USE_BEDROCK',
  'CLAUDE_CODE_USE_VERTEX',
  'CLAUDE_CODE_USE_FOUNDRY',
] as const;

const FALLBACK_SYSTEM = 'You are a helpful assistant.';

// ---------------------------------------------------------------------------
// Running the `claude` command
// ---------------------------------------------------------------------------

export interface CliRequest {
  /** System prompt text. */
  system: string;
  /** Everything else, sent over stdin. */
  prompt: string;
  /** When set, the reply is forced to this JSON Schema and returned as `structured`. */
  jsonSchema?: Record<string, unknown>;
  signal?: AbortSignal;
}

export interface CliResponse {
  text: string;
  structured?: unknown;
  usage?: { inputTokens: number; outputTokens: number };
}

export type CliRunner = (request: CliRequest) => Promise<CliResponse>;

/** The id `claude --model` expects: our routing prefix removed. */
export function cliModelName(model: string): string {
  return model.startsWith(CLAUDE_CODE_MODEL_PREFIX) ? model.slice(CLAUDE_CODE_MODEL_PREFIX.length) : model;
}

/** Locate the `claude` executable: CLAUDE_CODE_BIN, else the first match on PATH. */
export function findClaudeBinary(env: NodeJS.ProcessEnv = process.env): string {
  const override = env.CLAUDE_CODE_BIN;
  if (override) {
    if (existsSync(override)) return override;
    throw new Error(`[Claude Code] CLAUDE_CODE_BIN=${override} does not exist.`);
  }
  const names = process.platform === 'win32' ? ['claude.exe', 'claude'] : ['claude'];
  for (const dir of (env.PATH ?? env.Path ?? '').split(delimiter)) {
    if (!dir) continue;
    for (const name of names) {
      const candidate = join(dir, name);
      if (existsSync(candidate)) return candidate;
    }
  }
  throw new Error(
    '[Claude Code] The `claude` command was not found. Install Claude Code, run `claude` once to sign in with your Claude account, or set CLAUDE_CODE_BIN to its full path.',
  );
}

export function buildCliArgs(model: string, systemFile: string, jsonSchema?: Record<string, unknown>): string[] {
  return [
    '-p',
    '--safe-mode',
    '--model',
    cliModelName(model),
    '--system-prompt-file',
    systemFile,
    '--tools',
    '',
    '--no-session-persistence',
    '--disable-slash-commands',
    '--output-format',
    'json',
    ...(jsonSchema ? ['--json-schema', JSON.stringify(jsonSchema)] : []),
  ];
}

export function buildCliEnv(env: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const clean: NodeJS.ProcessEnv = { ...env };
  for (const name of BILLING_ENV_VARS) delete clean[name];
  return clean;
}

/** Parse `claude --output-format json`; throws a classifiable error when the run failed. */
export function parseCliOutput(stdout: string, exitCode: number | null, stderr: string): CliResponse {
  let data: Record<string, unknown> | undefined;
  const trimmed = stdout.trim();
  if (trimmed) {
    try {
      const parsed = JSON.parse(trimmed);
      data = Array.isArray(parsed)
        ? ([...parsed].reverse().find((e) => e && e.type === 'result') as Record<string, unknown> | undefined)
        : parsed;
    } catch {
      data = undefined;
    }
  }
  if (!data || data.type !== 'result') {
    const detail = (stderr || stdout).trim().slice(0, 500);
    throw new Error(`[Claude Code] claude exited with code ${exitCode} and no result${detail ? `: ${detail}` : ''}`);
  }
  const text = data.result;
  if (data.is_error || typeof text !== 'string') {
    const message = String(text ?? data.subtype ?? 'unknown error').slice(0, 500);
    // "Not logged in" must read as an auth problem so Dexter does not retry it.
    const hint = /log ?in|oauth|credential/i.test(message) ? ' (authentication: run `claude` and sign in)' : '';
    throw new Error(`[Claude Code] ${message}${hint}`);
  }
  const usage = data.usage as Record<string, number> | undefined;
  return {
    text,
    structured: data.structured_output,
    usage: usage
      ? {
          inputTokens:
            (usage.input_tokens ?? 0) + (usage.cache_creation_input_tokens ?? 0) + (usage.cache_read_input_tokens ?? 0),
          outputTokens: usage.output_tokens ?? 0,
        }
      : undefined,
  };
}

// Agent loops can fan out (subagents, parallel tools). A plan's rate limit is shared by
// every `claude` process, so keep the number running at once modest.
const MAX_PARALLEL = Math.max(1, Number.parseInt(process.env.DEXTER_CLAUDE_CONCURRENCY ?? '3', 10) || 3);
let running = 0;
const waiting: Array<() => void> = [];

async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (running >= MAX_PARALLEL) await new Promise<void>((resolve) => waiting.push(resolve));
  running += 1;
  try {
    return await fn();
  } finally {
    running -= 1;
    waiting.shift()?.();
  }
}

export function createCliRunner(model: string, timeoutMs: number, binary?: string): CliRunner {
  return (request) =>
    withSlot(async () => {
      const bin = binary ?? findClaudeBinary();
      const dir = await mkdtemp(join(tmpdir(), 'dexter-claude-'));
      try {
        const systemFile = join(dir, 'system.md');
        await writeFile(systemFile, request.system.trim() ? request.system : FALLBACK_SYSTEM, 'utf-8');
        const workdir = join(tmpdir(), 'dexter-claude-cwd');
        await mkdir(workdir, { recursive: true });
        return await new Promise<CliResponse>((resolve, reject) => {
          const child = spawn(bin, buildCliArgs(model, systemFile, request.jsonSchema), {
            cwd: workdir,
            env: buildCliEnv(),
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
          });
          let stdout = '';
          let stderr = '';
          let settled = false;
          const settle = (fn: () => void) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            request.signal?.removeEventListener('abort', onAbort);
            fn();
          };
          const onAbort = () => {
            child.kill();
            settle(() => reject(new Error('[Claude Code] call aborted')));
          };
          const timer = setTimeout(() => {
            child.kill();
            settle(() => reject(new Error(`[Claude Code] claude timed out after ${Math.round(timeoutMs / 1000)}s`)));
          }, timeoutMs);
          if (request.signal?.aborted) return onAbort();
          request.signal?.addEventListener('abort', onAbort, { once: true });

          child.stdout.setEncoding('utf-8').on('data', (chunk) => (stdout += chunk));
          child.stderr.setEncoding('utf-8').on('data', (chunk) => (stderr += chunk));
          child.on('error', (err) => settle(() => reject(new Error(`[Claude Code] could not start claude: ${err.message}`))));
          child.on('close', (code) =>
            settle(() => {
              try {
                resolve(parseCliOutput(stdout, code, stderr));
              } catch (err) {
                reject(err);
              }
            }),
          );
          child.stdin.on('error', () => undefined); // the child may exit before reading; its exit code explains why
          child.stdin.end(request.prompt, 'utf-8');
        });
      } finally {
        await rm(dir, { recursive: true, force: true }).catch(() => undefined);
      }
    });
}

// ---------------------------------------------------------------------------
// Messages -> prompt, reply -> messages
// ---------------------------------------------------------------------------

interface ToolDef {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === 'string') return block;
        if (block && typeof block === 'object' && (block as { type?: string }).type === 'text') {
          return String((block as { text?: unknown }).text ?? '');
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  return content == null ? '' : JSON.stringify(content);
}

/** System text (all system messages) and the rest of the conversation as a transcript. */
export function splitConversation(messages: BaseMessage[]): { system: string; transcript: string } {
  const system: string[] = [];
  const turns: string[] = [];
  for (const message of messages) {
    const type = message._getType();
    const text = textOf(message.content);
    if (type === 'system') {
      system.push(text);
    } else if (type === 'human') {
      turns.push(`### User\n${text}`);
    } else if (type === 'ai') {
      const calls = (message as AIMessage).tool_calls ?? [];
      const callText = calls
        .map((call) => `(called tool ${call.name} id=${call.id ?? ''} args=${JSON.stringify(call.args)})`)
        .join('\n');
      turns.push(`### Assistant\n${[text, callText].filter(Boolean).join('\n')}`);
    } else if (type === 'tool') {
      const tool = message as unknown as { name?: string; tool_call_id?: string };
      turns.push(`### Tool result (${tool.name ?? 'tool'} id=${tool.tool_call_id ?? ''})\n${text}`);
    } else {
      turns.push(`### ${type}\n${text}`);
    }
  }
  return { system: system.join('\n\n'), transcript: turns.join('\n\n') };
}

function toolProtocol(tools: ToolDef[]): string {
  const listing = tools
    .map((tool) => `- ${tool.name}: ${tool.description ?? ''}\n  parameters (JSON Schema): ${JSON.stringify(tool.parameters ?? {})}`)
    .join('\n');
  return [
    'You are the language model inside an agent loop. Reply with ONE JSON object and nothing else:',
    '{"content": string, "tool_calls": [{"name": string, "args": object}]}',
    '- To use tools, list one or more calls in tool_calls. Each args object must follow that tool\'s parameters exactly. Put any short reasoning worth showing in content. Tool results arrive in the next turn.',
    '- When you are done or need no tool, return "tool_calls": [] and put your complete final answer in content.',
    '- Only call the tools listed here:',
    listing,
  ].join('\n');
}

function toolReplySchema(tools: ToolDef[]): Record<string, unknown> {
  return {
    type: 'object',
    properties: {
      content: { type: 'string' },
      tool_calls: {
        type: 'array',
        items: {
          type: 'object',
          properties: { name: { type: 'string', enum: tools.map((t) => t.name) }, args: { type: 'object' } },
          required: ['name', 'args'],
        },
      },
    },
    required: ['content', 'tool_calls'],
  };
}

function newCallId(): string {
  return `call_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
}

/** Pull the first JSON object out of text (fenced, whole, or embedded). */
export function extractJsonObject(text: string): unknown {
  const fence = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
  for (const candidate of [fence?.[1], text.trim()]) {
    if (!candidate) continue;
    try {
      return JSON.parse(candidate);
    } catch {
      /* try the next shape */
    }
  }
  const start = text.indexOf('{');
  if (start !== -1) {
    let depth = 0;
    for (let i = start; i < text.length; i++) {
      if (text[i] === '{') depth += 1;
      else if (text[i] === '}' && --depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          break;
        }
      }
    }
  }
  throw new Error(`[Claude Code] no JSON object in the model reply: ${text.slice(0, 200)}`);
}

function toMessages(input: BaseLanguageModelInput): BaseMessage[] {
  if (typeof input === 'string') return [new HumanMessage(input)];
  if (Array.isArray(input)) return input.map((m) => coerceMessageLikeToMessage(m));
  return (input as { toChatMessages(): BaseMessage[] }).toChatMessages();
}

// ---------------------------------------------------------------------------
// The chat model
// ---------------------------------------------------------------------------

export interface ChatClaudeCodeCallOptions extends BaseChatModelCallOptions {
  tools?: ToolDef[];
}

export interface ChatClaudeCodeParams extends BaseChatModelParams {
  /** Value for `claude --model`: an alias (`sonnet`) or a full id (`claude-sonnet-5-5`); our `claude-code:` prefix is allowed. */
  model: string;
  /** Per-call limit. Default 5 minutes. */
  timeoutMs?: number;
  /** Replaces the real `claude` process (tests). */
  runner?: CliRunner;
}

export class ChatClaudeCode extends BaseChatModel<ChatClaudeCodeCallOptions> {
  model: string;
  private readonly runner: CliRunner;

  static lc_name() {
    return 'ChatClaudeCode';
  }

  constructor(params: ChatClaudeCodeParams) {
    super(params);
    this.model = params.model;
    this.runner = params.runner ?? createCliRunner(params.model, params.timeoutMs ?? 300_000);
  }

  _llmType() {
    return 'claude-code';
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  bindTools(tools: any[], kwargs?: Partial<ChatClaudeCodeCallOptions>): Runnable<BaseLanguageModelInput, AIMessageChunk, ChatClaudeCodeCallOptions> {
    const defs: ToolDef[] = tools.map((tool) => convertToOpenAITool(tool).function as ToolDef);
    return this.withConfig({ ...kwargs, tools: defs } as Partial<ChatClaudeCodeCallOptions>) as unknown as Runnable<
      BaseLanguageModelInput,
      AIMessageChunk,
      ChatClaudeCodeCallOptions
    >;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  withStructuredOutput(outputSchema: any, config?: { includeRaw?: boolean }): Runnable<BaseLanguageModelInput, any> {
    // zod emits a `$schema: …/draft/2020-12/…` marker that `claude --json-schema` refuses
    // ("no schema with key or ref"), so drop it; the rest of the schema is plain.
    const { $schema: _draft, ...jsonSchema } = toJsonSchema(outputSchema) as Record<string, unknown>;
    return new RunnableLambda({
      func: async (input: BaseLanguageModelInput, options?: { signal?: AbortSignal }) => {
        const { system, transcript } = splitConversation(toMessages(input));
        const response = await this.runner({
          system: `${system}\n\nReturn ONLY a JSON object that matches the required schema.`,
          prompt: transcript,
          jsonSchema,
          signal: options?.signal,
        });
        const object = response.structured ?? extractJsonObject(response.text);
        const parsed = typeof outputSchema?.parse === 'function' ? outputSchema.parse(object) : object;
        return config?.includeRaw ? { raw: new AIMessage(response.text), parsed } : parsed;
      },
    });
  }

  async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    _runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    const tools = (options as ChatClaudeCodeCallOptions).tools ?? [];
    const { system, transcript } = splitConversation(messages);
    const request: CliRequest = { system, prompt: transcript, signal: options.signal };

    if (tools.length > 0) {
      request.system = `${system}\n\n${toolProtocol(tools)}`;
      request.jsonSchema = toolReplySchema(tools);
      request.prompt = `${transcript}\n\n### Assistant (reply now as the JSON object described in the instructions)`;
    }

    const response = await this.runner(request);
    let content = response.text;
    let toolCalls: Array<{ id: string; name: string; args: Record<string, unknown>; type: 'tool_call' }> = [];

    if (tools.length > 0) {
      const reply = (response.structured ?? extractJsonObject(response.text)) as {
        content?: unknown;
        tool_calls?: Array<{ name?: unknown; args?: unknown }>;
      };
      content = typeof reply.content === 'string' ? reply.content : '';
      const known = new Set(tools.map((t) => t.name));
      toolCalls = (reply.tool_calls ?? [])
        .filter((call) => typeof call.name === 'string' && known.has(call.name))
        .map((call) => ({
          id: newCallId(),
          name: call.name as string,
          args: (call.args && typeof call.args === 'object' ? call.args : {}) as Record<string, unknown>,
          type: 'tool_call' as const,
        }));
    }

    const usage = response.usage
      ? {
          input_tokens: response.usage.inputTokens,
          output_tokens: response.usage.outputTokens,
          total_tokens: response.usage.inputTokens + response.usage.outputTokens,
        }
      : undefined;
    const message = new AIMessage({ content, tool_calls: toolCalls, usage_metadata: usage });
    return { generations: [{ text: content, message }], llmOutput: { tokenUsage: usage } };
  }

  /** Streams the whole reply as one chunk: the CLI answers in a single JSON document. */
  async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatGenerationChunk> {
    const result = await this._generate(messages, options, runManager);
    const full = result.generations[0].message as AIMessage;
    const chunk = new AIMessageChunk({
      content: full.content,
      tool_call_chunks: (full.tool_calls ?? []).map((call, index) => ({
        name: call.name,
        args: JSON.stringify(call.args),
        id: call.id,
        index,
        type: 'tool_call_chunk' as const,
      })),
      usage_metadata: full.usage_metadata,
    });
    yield new ChatGenerationChunk({ message: chunk, text: typeof full.content === 'string' ? full.content : '' });
    await runManager?.handleLLMNewToken(typeof full.content === 'string' ? full.content : '');
  }
}
