#!/usr/bin/env bun
/**
 * Live check of the "Claude Code (your subscription)" provider: three real calls through
 * the local `claude` command (plain text, tool calling, structured output).
 *
 *   bun run check:claude [model]        e.g.  bun run check:claude claude-code:claude-haiku-5-5
 *
 * Uses a little of your Claude plan. Needs no API key.
 */
import { HumanMessage, SystemMessage, type AIMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { z } from 'zod';
import { getChatModel } from '../src/model/llm.js';
import { findClaudeBinary } from '../src/model/claude-code.js';

const model = process.argv[2] ?? 'claude-code:claude-sonnet-5-5';
console.log(`binary : ${findClaudeBinary()}`);
console.log(`model  : ${model}\n`);

async function timed<T>(label: string, fn: () => Promise<T>): Promise<T> {
  const start = Date.now();
  const result = await fn();
  console.log(`  (${label}: ${((Date.now() - start) / 1000).toFixed(1)}s)`);
  return result;
}

let failed = false;
const check = (name: string, ok: boolean, detail: unknown) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
  if (!ok) failed = true;
};

const llm = getChatModel(model);

// 1. plain text
const plain = await timed('plain', () =>
  llm.invoke([new SystemMessage('You are a terse test responder.'), new HumanMessage('Reply with only the single word PONG.')]),
);
check('plain text', String(plain.content).toUpperCase().includes('PONG'), String(plain.content));

// 2. tool calling (the agent loop's core need)
const getIncomeStatement = tool(async () => 'unused', {
  name: 'get_income_statement',
  description: 'Fetch a company income statement',
  schema: z.object({ ticker: z.string().describe('Ticker symbol'), period: z.enum(['annual', 'quarterly']) }),
});
const webSearch = tool(async () => 'unused', {
  name: 'web_search',
  description: 'Search the web',
  schema: z.object({ query: z.string() }),
});
const withTools = llm.bindTools!([getIncomeStatement, webSearch]);
const toolTurn = (await timed('tools', () =>
  withTools.invoke([
    new SystemMessage('You are a financial research agent. Use the tools to get data.'),
    new HumanMessage("What was Apple's last full-year revenue? Fetch the data."),
  ]),
)) as AIMessage;
const call = toolTurn.tool_calls?.[0];
check('tool call', call?.name === 'get_income_statement' && /aapl/i.test(String(call?.args?.ticker)), call ?? 'no tool call');

// 3. structured output
const Verdict = z.object({ verdict: z.enum(['buy', 'hold', 'sell']), reason: z.string() });
const structured = await timed('structured', () =>
  llm.withStructuredOutput!(Verdict).invoke([new HumanMessage('A company trades at 8x earnings with 25% ROE and almost no debt. Give a verdict.')]),
);
check('structured output', Verdict.safeParse(structured).success, structured);

process.exit(failed ? 1 : 0);
