import { describe, expect, test } from 'bun:test';
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { checkApiKeyExistsForProvider } from '../utils/env.js';
import { getModelsForProvider } from '../utils/model.js';
import { resolveProvider } from '../providers.js';
import {
  BILLING_ENV_VARS,
  ChatClaudeCode,
  buildCliArgs,
  buildCliEnv,
  cliModelName,
  extractJsonObject,
  parseCliOutput,
  splitConversation,
  type CliRequest,
  type CliResponse,
} from './claude-code.js';

/** A runner that records the request and returns a canned reply, so no `claude` process starts. */
function fakeRunner(reply: Partial<CliResponse>) {
  const requests: CliRequest[] = [];
  const runner = async (request: CliRequest): Promise<CliResponse> => {
    requests.push(request);
    return { text: '', ...reply };
  };
  return { runner, requests };
}

const getIncome = tool(async () => 'ok', {
  name: 'get_income_statement',
  description: 'Fetch an income statement',
  schema: z.object({ ticker: z.string(), period: z.enum(['annual', 'quarterly']) }),
});
const webSearch = tool(async () => 'ok', {
  name: 'web_search',
  description: 'Search the web',
  schema: z.object({ query: z.string() }),
});

describe('provider wiring', () => {
  test('claude-code: models route to Claude Code, not Anthropic', () => {
    expect(resolveProvider('claude-code:claude-sonnet-5-5').id).toBe('claude-code');
    expect(resolveProvider('claude-opus-5-5').id).toBe('anthropic');
  });

  test('needs no API key, and offers the subscription models', () => {
    expect(checkApiKeyExistsForProvider('claude-code')).toBe(true);
    const ids = getModelsForProvider('claude-code').map((m) => m.id);
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.every((id) => id.startsWith('claude-code:'))).toBe(true);
  });

  test('cliModelName strips only our routing prefix', () => {
    expect(cliModelName('claude-code:claude-sonnet-5-5')).toBe('claude-sonnet-5-5');
    expect(cliModelName('claude-code:opus')).toBe('opus');
    expect(cliModelName('sonnet')).toBe('sonnet');
  });
});

describe('the command we run', () => {
  test('uses the subscription login: safe-mode, no tools, never --bare', () => {
    const args = buildCliArgs('claude-code:claude-sonnet-5-5', '/tmp/system.md');
    expect(args).toContain('--safe-mode');
    expect(args).not.toContain('--bare');
    expect(args[args.indexOf('--model') + 1]).toBe('claude-sonnet-5-5');
    expect(args[args.indexOf('--tools') + 1]).toBe('');
    expect(args).toContain('--no-session-persistence');
    expect(args[args.indexOf('--system-prompt-file') + 1]).toBe('/tmp/system.md');
    expect(args).not.toContain('--json-schema');
  });

  test('passes a JSON schema through when asked', () => {
    const args = buildCliArgs('claude-code:opus', 'f', { type: 'object' });
    expect(JSON.parse(args[args.indexOf('--json-schema') + 1])).toEqual({ type: 'object' });
  });

  test('API-billing variables never reach the child', () => {
    const env: NodeJS.ProcessEnv = { KEEP_ME: 'yes' };
    for (const name of BILLING_ENV_VARS) env[name] = 'set-by-a-test';
    const clean = buildCliEnv(env);
    for (const name of BILLING_ENV_VARS) expect(clean[name]).toBeUndefined();
    expect(clean.KEEP_ME).toBe('yes');
  });
});

describe('reading claude output', () => {
  const ok = (extra: object = {}) => JSON.stringify({ type: 'result', is_error: false, result: 'hi', ...extra });

  test('returns text, structured output and token usage', () => {
    const parsed = parseCliOutput(
      ok({ structured_output: { a: 1 }, usage: { input_tokens: 2, cache_creation_input_tokens: 10, cache_read_input_tokens: 3, output_tokens: 5 } }),
      0,
      '',
    );
    expect(parsed.text).toBe('hi');
    expect(parsed.structured).toEqual({ a: 1 });
    expect(parsed.usage).toEqual({ inputTokens: 15, outputTokens: 5 });
  });

  test('an error result throws, and a login problem reads as authentication', () => {
    const notLoggedIn = JSON.stringify({ type: 'result', is_error: true, result: 'Not logged in · Please run /login' });
    expect(() => parseCliOutput(notLoggedIn, 1, '')).toThrow(/authentication/);
    expect(() => parseCliOutput(JSON.stringify({ type: 'result', is_error: true, result: 'boom' }), 1, '')).toThrow(/boom/);
  });

  test('no result at all throws with the exit code and stderr', () => {
    expect(() => parseCliOutput('', 1, 'it died')).toThrow(/code 1.*it died/);
    expect(() => parseCliOutput('not json', 0, '')).toThrow(/no result/);
  });

  test('extractJsonObject finds fenced, bare and embedded objects', () => {
    expect(extractJsonObject('```json\n{"a":1}\n```')).toEqual({ a: 1 });
    expect(extractJsonObject('{"a":2}')).toEqual({ a: 2 });
    expect(extractJsonObject('Sure! {"a":3} hope that helps')).toEqual({ a: 3 });
    expect(() => extractJsonObject('nothing here')).toThrow(/no JSON/);
  });
});

describe('conversation -> prompt', () => {
  test('system text is separated; tool calls and results stay in the transcript', () => {
    const { system, transcript } = splitConversation([
      new SystemMessage('Be brief.'),
      new HumanMessage('Apple revenue?'),
      new AIMessage({ content: 'Looking.', tool_calls: [{ id: 'c1', name: 'get_income_statement', args: { ticker: 'AAPL' }, type: 'tool_call' }] }),
      new ToolMessage({ content: '{"revenue":1}', tool_call_id: 'c1', name: 'get_income_statement' }),
    ]);
    expect(system).toBe('Be brief.');
    expect(transcript).toContain('### User\nApple revenue?');
    expect(transcript).toContain('(called tool get_income_statement id=c1');
    expect(transcript).toContain('### Tool result (get_income_statement id=c1)\n{"revenue":1}');
  });
});

describe('ChatClaudeCode', () => {
  test('plain call returns the text and usage', async () => {
    const { runner, requests } = fakeRunner({ text: 'Hello!', usage: { inputTokens: 7, outputTokens: 3 } });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const result = await model.invoke([new SystemMessage('sys'), new HumanMessage('hi')]);
    expect(result.content).toBe('Hello!');
    expect(result.usage_metadata?.total_tokens).toBe(10);
    expect(requests[0].system).toBe('sys');
    expect(requests[0].jsonSchema).toBeUndefined();
  });

  test('bindTools turns the JSON reply into tool_calls and drops unknown tools', async () => {
    const { runner, requests } = fakeRunner({
      text: '',
      structured: {
        content: 'Fetching.',
        tool_calls: [
          { name: 'get_income_statement', args: { ticker: 'AAPL', period: 'annual' } },
          { name: 'not_a_tool', args: {} },
        ],
      },
    });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const result = await model.bindTools([getIncome, webSearch]).invoke([new HumanMessage('Apple revenue?')]);

    const calls = (result as AIMessage).tool_calls ?? [];
    expect(calls).toHaveLength(1);
    expect(calls[0].name).toBe('get_income_statement');
    expect(calls[0].args).toEqual({ ticker: 'AAPL', period: 'annual' });
    expect(calls[0].id).toBeTruthy();
    expect(result.content).toBe('Fetching.');

    // The tools are described to the model, and the reply shape is enforced.
    expect(requests[0].system).toContain('get_income_statement');
    expect(requests[0].system).toContain('web_search');
    const enumNames = (requests[0].jsonSchema as any).properties.tool_calls.items.properties.name.enum;
    expect(enumNames).toEqual(['get_income_statement', 'web_search']);
  });

  test('no tool calls means the content is the final answer', async () => {
    const { runner } = fakeRunner({ structured: { content: 'Revenue was $1.', tool_calls: [] } });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const result = (await model.bindTools([getIncome]).invoke([new HumanMessage('q')])) as AIMessage;
    expect(result.tool_calls).toHaveLength(0);
    expect(result.content).toBe('Revenue was $1.');
  });

  test('a reply that is only text JSON (no structured_output) still parses', async () => {
    const { runner } = fakeRunner({ text: '```json\n{"content":"x","tool_calls":[{"name":"web_search","args":{"query":"q"}}]}\n```' });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const result = (await model.bindTools([webSearch]).invoke([new HumanMessage('q')])) as AIMessage;
    expect(result.tool_calls?.[0].name).toBe('web_search');
  });

  test('withStructuredOutput validates against the zod schema', async () => {
    const schema = z.object({ verdict: z.enum(['buy', 'hold']), score: z.number() });
    const { runner, requests } = fakeRunner({ structured: { verdict: 'hold', score: 3 } });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const result = await model.withStructuredOutput(schema).invoke([new SystemMessage('judge'), new HumanMessage('x')]);
    expect(result).toEqual({ verdict: 'hold', score: 3 });
    expect((requests[0].jsonSchema as any).properties.verdict).toBeDefined();
    // `claude --json-schema` rejects zod's draft-2020-12 `$schema` marker (found by the live check).
    expect((requests[0].jsonSchema as any).$schema).toBeUndefined();

    const bad = fakeRunner({ structured: { verdict: 'sell', score: 'high' } });
    const strict = new ChatClaudeCode({ model: 'claude-code:opus', runner: bad.runner });
    await expect(strict.withStructuredOutput(schema).invoke([new HumanMessage('x')])).rejects.toThrow();
  });

  test('streaming yields the whole reply, with tool calls, as one chunk', async () => {
    const { runner } = fakeRunner({ structured: { content: 'go', tool_calls: [{ name: 'web_search', args: { query: 'q' } }] } });
    const model = new ChatClaudeCode({ model: 'claude-code:opus', runner });
    const chunks = [];
    for await (const chunk of await model.bindTools([webSearch]).stream([new HumanMessage('q')])) chunks.push(chunk);
    expect(chunks.length).toBeGreaterThan(0);
    const calls = chunks.flatMap((c) => (c as any).tool_calls ?? []);
    expect(calls[0].name).toBe('web_search');
    expect(calls[0].args).toEqual({ query: 'q' });
  });

  test('runner failures propagate', async () => {
    const model = new ChatClaudeCode({
      model: 'claude-code:opus',
      runner: async () => {
        throw new Error('[Claude Code] claude timed out after 1s');
      },
    });
    await expect(model.invoke([new HumanMessage('hi')])).rejects.toThrow(/timed out/);
  });
});
