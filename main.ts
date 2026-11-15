import * as http from 'node:http';
import * as fs from 'node:fs/promises';
import { setTimeout as delay } from 'node:timers/promises';

type Obj = Record<string, any>;
type Provider = { id: string; path: string; wire: string; auth: string; key_env: string; key: string };
type Rating = { provider: string; model: string; rating_milli: number; context_length: number; tools: boolean; vision: boolean; json: boolean };
type ManifestEntry = { id: string; api: string; stream: boolean; weight: number; requires: Obj };
type Model = { id: string; free: boolean; context_length: number };

const SCHEMAS = ['free-model-aggregator-01/providers/v1', 'free-model-aggregator-01/ratings/v1', 'free-model-aggregator-01/workload/v1'];
const CREATED = 1767225600;
let directory: Provider[] = [];
let ratings: Rating[] = [];
let manifest: ManifestEntry[] = [];
let catalog: Obj = { providers: [] };
let requestIndex = 0;
const failures = new Map<string, number>();

function headersFor(p: Provider): Record<string, string> {
  const key = p.key || process.env[p.key_env] || '';
  if (p.auth === 'bearer') return { authorization: 'Bearer ' + key };
  if (p.auth === 'x-api-key') return { 'x-api-key': key };
  if (p.auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function json(res: http.ServerResponse, status: number, value: unknown): void {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(value));
}

function scanDocuments(text: string): Obj[] {
  const out: Obj[] = [];
  let start = -1, depth = 0, quoted = false, escaped = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (start < 0) { if (c === '{') { start = i; depth = 1; } continue; }
    if (quoted) { if (escaped) escaped = false; else if (c === String.fromCharCode(92)) escaped = true; else if (c === '"') quoted = false; continue; }
    if (c === '"') quoted = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) {
      try { const v = JSON.parse(text.slice(start, i + 1)); if (SCHEMAS.includes(v.schema)) out.push(v); } catch {}
      start = -1;
    }
  }
  return out;
}

async function readBrief(): Promise<void> {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (!path) return;
  const docs = scanDocuments(await fs.readFile(path, 'utf8'));
  for (const d of docs) {
    if (d.schema === SCHEMAS[0]) directory = d.providers || [];
    if (d.schema === SCHEMAS[1]) ratings = d.ratings || [];
    if (d.schema === SCHEMAS[2]) manifest = d.manifest || [];
  }
}

async function buildCatalog(): Promise<void> {
  const providers: Obj[] = [];
  for (const p of directory) {
    const route = p.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
    let models: Model[] = [];
    if (process.env.MAKEFUN_ENV_URL) {
      try {
        const response = await fetch(providerUrl(p, route), { headers: headersFor(p), signal: AbortSignal.timeout(5000) });
        if (!response.ok) throw new Error('catalog status ' + response.status);
        const body: Obj = await response.json();
        const rows = p.wire === 'gemini' ? body.models || [] : body.data || [];
        models = rows.map((m: Obj) => ({ id: p.wire === 'gemini' ? String(m.name || '').replace(/^models\//, '') : m.id, free: !!m.free, context_length: m.context_length })).filter((m: Model) => !!m.id);
      } catch {
        models = ratings.filter(r => r.provider === p.id).map(r => ({ id: r.model, free: true, context_length: r.context_length }));
      }
    } else {
      models = ratings.filter(r => r.provider === p.id).map(r => ({ id: r.model, free: true, context_length: r.context_length }));
    }
    const seen = new Set<string>();
    models = models.filter(m => { if (seen.has(m.id)) return false; seen.add(m.id); return true; });
    providers.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  }
  catalog = { providers };
}

function collectText(content: any): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.map(x => typeof x === 'string' ? x : x?.text || '').join('');
}

function bodyRequirements(body: Obj, api: string): Obj {
  const messages = body.messages || [];
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  const vision = messages.some((m: Obj) => Array.isArray(m.content) && m.content.some((b: Obj) => b.type === 'image' || b.type === 'image_url' || b.inlineData || b.fileData));
  const json = api === 'openai' && ['json_object', 'json_schema'].includes(body.response_format?.type);
  return { tools, vision, json, min_context: 0 };
}

function requirements(body: Obj, api: string, index: number): Obj {
  const a = bodyRequirements(body, api), b = manifest[index]?.requires || {};
  return { tools: !!a.tools || !!b.tools, vision: !!a.vision || !!b.vision, json: !!a.json || !!b.json, min_context: Math.max(a.min_context, b.min_context || 0) };
}

function textFromBlocks(value: any): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value.map((x: Obj) => x.text || '').join('');
}

function messagesFor(body: Obj, client: string): Obj[] {
  if (client === 'openai') return body.messages || [];
  const out: Obj[] = [];
  const sys = textFromBlocks(body.system);
  if (sys) out.push({ role: 'system', content: sys });
  for (const m of body.messages || []) {
    const content = Array.isArray(m.content) ? m.content : [{ type: 'text', text: m.content }];
    const chunks: Obj[] = [];
    for (const b of content) {
      if (b.type === 'text') chunks.push({ type: 'text', text: b.text });
      else if (b.type === 'image') {
        const s = b.source || {};
        const url = s.type === 'base64' ? 'data:' + (s.media_type || 'image/png') + ';base64,' + s.data : (s.url || '');
        chunks.push({ type: 'image_url', image_url: { url } });
      } else if (b.type === 'tool_use') chunks.push({ type: 'text', text: JSON.stringify(b.input || {}) });
      else if (b.type === 'tool_result') chunks.push({ type: 'text', text: textFromBlocks(b.content) });
    }
    out.push({ role: m.role === 'assistant' ? 'assistant' : 'user', content: chunks });
  }
  return out;
}

function systemText(body: Obj, client: string): string {
  if (client === 'anthropic') return textFromBlocks(body.system);
  return (body.messages || []).filter((m: Obj) => m.role === 'system').map((m: Obj) => textFromBlocks(m.content)).join('\n');
}

function toolsFor(body: Obj, client: string): Obj[] {
  if (client === 'openai') return body.tools || [];
  return (body.tools || []).map((t: Obj) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }));
}

function toGemini(body: Obj, client: string): Obj {
  const sourceMessages = body.messages || [];
  const sys = systemText(body, client);
  const contents = sourceMessages.filter((m: Obj) => m.role !== 'system').map((m: Obj) => {
    const content = Array.isArray(m.content) ? m.content : [{ type: 'text', text: m.content }];
    const parts: Obj[] = [];
    for (const block of content) {
      if (block.type === 'text') parts.push({ text: block.text });
      else if (block.type === 'image_url') {
        const url = typeof block.image_url === 'string' ? block.image_url : block.image_url?.url || '';
        const image = geminiImage(url);
        if (image) parts.push(image);
      } else if (block.type === 'image') {
        const source = block.source || {};
        if (source.type === 'base64') parts.push({ inlineData: { mimeType: source.media_type || 'image/png', data: source.data } });
        else if (source.type === 'url' && source.url) parts.push({ fileData: { fileUri: source.url } });
      } else if (block.inlineData) parts.push({ inlineData: block.inlineData });
      else if (block.fileData) parts.push({ fileData: block.fileData });
    }
    return { role: m.role === 'assistant' ? 'model' : 'user', parts };
  });
  const result: Obj = { contents };
  if (sys) result.systemInstruction = { parts: [{ text: sys }] };
  const ts = toolsFor(body, client);
  if (ts.length) result.tools = [{ functionDeclarations: ts.map((t: Obj) => t.function || t) }];
  const cfg: Obj = {};
  const max = body.max_tokens ?? body.max_completion_tokens;
  if (max !== undefined) cfg.maxOutputTokens = max;
  const stop = body.stop ?? body.stop_sequences;
  if (stop !== undefined) cfg.stopSequences = Array.isArray(stop) ? stop : [stop];
  if (body.temperature !== undefined) cfg.temperature = body.temperature;
  if (body.response_format?.type === 'json_object' || body.response_format?.type === 'json_schema') cfg.responseMimeType = 'application/json';
  if (Object.keys(cfg).length) result.generationConfig = cfg;
  return result;
}

function geminiImage(url: string): Obj | undefined {
  const match = /^data:([^;,]+)?;base64,(.*)$/.exec(url);
  if (match) return { inlineData: { mimeType: match[1] || 'image/png', data: match[2] } };
  if (url) return { fileData: { fileUri: url } };
  return undefined;
}

function toOpenAI(body: Obj, client: string, model: string): Obj {
  if (client === 'openai') {
    const out: Obj = { ...body, model };
    if (body.stream) out.stream_options = { ...(body.stream_options || {}), include_usage: true };
    return out;
  }
  const msgs = messagesFor(body, client);
  const ts = (body.tools || []).map((t: Obj) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.input_schema } }));
  const out: Obj = { model, messages: msgs, max_tokens: body.max_tokens };
  if (ts.length) out.tools = ts;
  if (body.stop_sequences !== undefined) out.stop = body.stop_sequences;
  for (const k of ['temperature', 'stream']) if (body[k] !== undefined) out[k] = body[k];
  if (body.stream) out.stream_options = { ...(body.stream_options || {}), include_usage: true };
  return out;
}

function toAnthropic(body: Obj, client: string, model: string): Obj {
  if (client === 'anthropic') return { ...body, model };
  const msgs = (body.messages || []).filter((m: Obj) => m.role !== 'system').map((m: Obj) => {
    if (typeof m.content === 'string') return { ...m };
    const content = Array.isArray(m.content) ? m.content.map((block: Obj) => {
      if (block.type === 'text') return block;
      if (block.type === 'image_url') {
        const url = typeof block.image_url === 'string' ? block.image_url : block.image_url?.url || '';
        const match = /^data:([^;,]+)?;base64,(.*)$/.exec(url);
        if (match) return { type: 'image', source: { type: 'base64', media_type: match[1] || 'image/png', data: match[2] } };
        return { type: 'image', source: { type: 'url', url } };
      }
      return block;
    }) : m.content;
    return { ...m, content };
  });
  const sys = systemText(body, client);
  const out: Obj = { model, messages: msgs, max_tokens: body.max_tokens ?? body.max_completion_tokens };
  if (sys) out.system = sys;
  if (body.tools?.length) out.tools = body.tools.map((t: Obj) => t.function ? { name: t.function.name, description: t.function.description, input_schema: t.function.parameters } : t);
  if (body.stop !== undefined) out.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop];
  for (const k of ['temperature', 'stream']) if (body[k] !== undefined) out[k] = body[k];
  return out;
}

function providerUrl(p: Provider, route: string): string {
  const env = (process.env.MAKEFUN_ENV_URL || '').replace(/\/$/, '');
  return env + p.path + route;
}

function requestUrl(p: Provider, model: string, wire: string, stream: boolean): string {
  if (wire === 'gemini') return providerUrl(p, '/v1beta/models/' + encodeURIComponent(model) + ':' + (stream ? 'streamGenerateContent' : 'generateContent'));
  return providerUrl(p, wire === 'anthropic' ? '/v1/messages' : '/v1/chat/completions');
}

function extractAnswer(p: Provider, data: any, client: string): { text: string; input: number; output: number } {
  if (p.wire === 'openai') {
    const c = data.choices?.[0] || {};
    return { text: c.message?.content || '', input: data.usage?.prompt_tokens || 0, output: data.usage?.completion_tokens || 0 };
  }
  if (p.wire === 'anthropic') return { text: textFromBlocks(data.content), input: data.usage?.input_tokens || 0, output: data.usage?.output_tokens || 0 };
  const c = data.candidates?.[0] || {};
  return { text: (c.content?.parts || []).map((x: Obj) => x.text || '').join(''), input: data.usageMetadata?.promptTokenCount || 0, output: data.usageMetadata?.candidatesTokenCount || 0 };
}

function clientError(api: string, message: string): Obj {
  return api === 'openai' ? { error: { message, type: 'api_error', param: null, code: 'api_error' } } : { type: 'error', error: { type: 'api_error', message } };
}

function clientReply(api: string, stream: boolean, model: string, text: string, input: number, output: number, res: http.ServerResponse): void {
  if (!stream) {
    if (api === 'openai') return json(res, 200, { id: 'chatcmpl-sim', object: 'chat.completion', created: CREATED, model, choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop' }], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } });
    return json(res, 200, { id: 'msg_sim', type: 'message', role: 'assistant', content: [{ type: 'text', text }], model, stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: input, output_tokens: output } });
  }
  res.statusCode = 200;
  res.setHeader('content-type', 'text/event-stream');
  res.setHeader('cache-control', 'no-cache');
  const pieces: string[] = [];
  for (let i = 0; i < text.length; i += 13) pieces.push(text.slice(i, i + 13));
  if (api === 'openai') {
    const chunk = (choices: any[], usage?: Obj) => ({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: CREATED, model, choices, ...(usage ? { usage } : {}) });
    res.write('data: ' + JSON.stringify(chunk([{ index: 0, delta: { role: 'assistant' }, finish_reason: null }])) + '\n\n');
    for (const part of pieces) res.write('data: ' + JSON.stringify(chunk([{ index: 0, delta: { content: part }, finish_reason: null }])) + '\n\n');
    res.write('data: ' + JSON.stringify(chunk([{ index: 0, delta: {}, finish_reason: 'stop' }])) + '\n\n');
    res.write('data: ' + JSON.stringify(chunk([], { prompt_tokens: input, completion_tokens: output, total_tokens: input + output })) + '\n\n');
    res.end('data: [DONE]\n\n');
  } else {
    const event = (name: string, value: Obj) => res.write('event: ' + name + '\ndata: ' + JSON.stringify(value) + '\n\n');
    event('message_start', { type: 'message_start', message: { id: 'msg_sim', type: 'message', role: 'assistant', content: [], model, stop_reason: null, stop_sequence: null, usage: { input_tokens: input, output_tokens: 0 } } });
    event('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
    for (const part of pieces) event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part } });
    event('content_block_stop', { type: 'content_block_stop', index: 0 });
    event('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: output } });
    event('message_stop', { type: 'message_stop' });
    res.end();
  }
}

async function readBody(req: http.IncomingMessage): Promise<Obj> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 10 * 1024 * 1024) throw Object.assign(new Error('Request body too large'), { status: 413 });
    chunks.push(Buffer.from(chunk));
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

async function inference(req: http.IncomingMessage, res: http.ServerResponse, api: string): Promise<void> {
  const index = requestIndex++;
  let body: Obj;
  try { body = await readBody(req); } catch (e: any) { return json(res, e.status || 400, clientError(api, 'Malformed request body')); }
  const reqs = requirements(body, api, index);
  const candidates = ratings.filter(r => {
    const p = directory.find(x => x.id === r.provider);
    return !!p && (!reqs.tools || r.tools) && (!reqs.vision || r.vision) && (!reqs.json || r.json) && r.context_length >= reqs.min_context && !(api === 'openai' && reqs.json && p.wire === 'anthropic');
  }).sort((a, b) => b.rating_milli - a.rating_milli || directory.findIndex(p => p.id === a.provider) - directory.findIndex(p => p.id === b.provider) || a.model.localeCompare(b.model));
  const seen = new Set<string>();
  const ordered = candidates.filter(c => { if (seen.has(c.provider)) return false; seen.add(c.provider); return true; }).sort((a, b) => (failures.get(a.provider) >= 2 ? 1 : 0) - (failures.get(b.provider) >= 2 ? 1 : 0) || candidates.indexOf(a) - candidates.indexOf(b));
  let chosen: { p: Provider; rating: Rating; answer: any } | undefined;
  let lastWait = 0, lastProvider: Provider | undefined, lastRating: Rating | undefined;
  for (const r of ordered) {
    const p = directory.find(x => x.id === r.provider)!;
    lastProvider = p; lastRating = r;
    try {
      const wireBody = p.wire === 'gemini' ? toGemini(body, api) : p.wire === 'openai' ? toOpenAI(body, api, r.model) : toAnthropic(body, api, r.model);
      const h: Record<string, string> = { ...headersFor(p), 'content-type': 'application/json' };
      if (p.wire === 'anthropic') h['anthropic-version'] = req.headers['anthropic-version']?.toString() || '2023-06-01';
      const response = await fetch(requestUrl(p, r.model, p.wire, !!body.stream), { method: 'POST', headers: h, body: JSON.stringify(wireBody), signal: AbortSignal.timeout(5000) });
      const raw = await response.text();
      if (!response.ok) {
        let err: any = {}; try { err = JSON.parse(raw); } catch {}
        const wait = Number(response.headers.get('retry-after') || err.retry_after || 0);
        process.stderr.write('Refusal ' + p.id + ' ' + response.status + '\n');
        if (wait) { lastWait = wait; process.stderr.write('Retry after ' + wait + '\n'); }
        failures.set(p.id, (failures.get(p.id) || 0) + 1);
        continue;
      }
      const data = parseProviderReply(raw, response.headers.get('content-type') || '', p.wire);
      failures.set(p.id, 0);
      chosen = { p, rating: r, answer: extractAnswer(p, data, api) };
      break;
    } catch (e) {
      process.stderr.write('Refusal ' + p.id + ' network or timeout\n');
      failures.set(p.id, (failures.get(p.id) || 0) + 1);
    }
  }
  if (!chosen && lastProvider && lastRating && lastWait > 0) {
    await delay(Math.min(lastWait, 2) * 1000);
    try {
      const p = lastProvider, r = lastRating;
      const wb = p.wire === 'gemini' ? toGemini(body, api) : p.wire === 'openai' ? toOpenAI(body, api, r.model) : toAnthropic(body, api, r.model);
      const h: Record<string, string> = { ...headersFor(p), 'content-type': 'application/json' };
      if (p.wire === 'anthropic') h['anthropic-version'] = req.headers['anthropic-version']?.toString() || '2023-06-01';
      const rr = await fetch(requestUrl(p, r.model, p.wire, !!body.stream), { method: 'POST', headers: h, body: JSON.stringify(wb), signal: AbortSignal.timeout(5000) });
      if (rr.ok) { const raw = await rr.text(); const data = parseProviderReply(raw, rr.headers.get('content-type') || '', p.wire); chosen = { p, rating: r, answer: extractAnswer(p, data, api) }; failures.set(p.id, 0); }
    } catch {}
  }
  if (!chosen) return json(res, 502, clientError(api, 'All providers refused the request'));
  clientReply(api, !!body.stream, chosen.rating.model, chosen.answer.text, chosen.answer.input, chosen.answer.output, res);
}

function parseStream(raw: string, wire: string): Obj {
  const frames = raw.split(/\n\n+/).map(x => x.split('\n').filter(l => l.startsWith('data:')).map(l => l.slice(5).trim()).join('')).filter(Boolean);
  const parsed = frames.filter(x => x !== '[DONE]').map(x => { try { return JSON.parse(x); } catch { return {}; } });
  if (wire === 'openai') {
    let text = '', usage: Obj = {};
    for (const f of parsed) { text += f.choices?.[0]?.delta?.content || ''; if (f.usage) usage = f.usage; }
    return { choices: [{ message: { content: text } }], usage };
  }
  if (wire === 'anthropic') {
    let text = '', input = 0, output = 0;
    for (const f of parsed) { text += f.delta?.text || ''; input = f.message?.usage?.input_tokens || input; output = f.usage?.output_tokens || output; }
    return { content: [{ text }], usage: { input_tokens: input, output_tokens: output } };
  }
  let text = '';
  let usageMetadata: Obj = {};
  for (const frame of parsed) {
    const parts = frame.candidates?.[0]?.content?.parts || [];
    text += parts.map((part: Obj) => part.text || '').join('');
    if (frame.usageMetadata) usageMetadata = frame.usageMetadata;
  }
  return { candidates: [{ content: { parts: [{ text }] } }], usageMetadata };
}

function parseProviderReply(raw: string, contentType: string, wire: string): Obj {
  const streamed = contentType.toLowerCase().includes('text/event-stream') || raw.split(/\r?\n/).some(line => line.startsWith('data:') || line.startsWith('event:'));
  return streamed ? parseStream(raw, wire) : JSON.parse(raw);
}

function uniqueModelIds(): string[] {
  const ids: string[] = [], seen = new Set<string>();
  for (const p of catalog.providers) for (const m of p.models) if (m.free && !seen.has(m.id)) { seen.add(m.id); ids.push(m.id); }
  return ids;
}

async function handler(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
  try {
    const path = new URL(req.url || '/', 'http://localhost').pathname;
    if (req.method === 'GET' && path === '/v1/models') {
      const ids = uniqueModelIds();
      if (req.headers['x-api-key'] !== undefined) return json(res, 200, { data: ids.map(id => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z' })), has_more: false });
      return json(res, 200, { object: 'list', data: ids.map(id => ({ id, object: 'model', created: CREATED, owned_by: 'aggregator' })) });
    }
    if (req.method === 'POST' && path === '/v1/chat/completions') return await inference(req, res, 'openai');
    if (req.method === 'POST' && path === '/v1/messages') return await inference(req, res, 'anthropic');
    json(res, 404, { error: { message: 'Not found' } });
  } catch (e) { if (!res.headersSent) json(res, 500, { error: { message: 'Request failed' } }); else res.end(); }
}

async function main(): Promise<void> {
  if (process.env.MAKEFUN_PROMPT_FILE) await readBrief();
  if (process.env.MAKEFUN_ENV_URL || process.env.MAKEFUN_PROMPT_FILE) await buildCatalog();
  if (process.argv.includes('--list')) {
    await new Promise<void>(resolve => process.stdout.write(JSON.stringify(catalog) + '\n', () => resolve()));
    return;
  }
  const env = process.env.MAKEFUN_ENV_URL;
  if (!env) throw new Error('MAKEFUN_ENV_URL is required');
  const server = http.createServer((req, res) => { void handler(req, res); });
  await new Promise<void>(resolve => server.listen(0, '0.0.0.0', resolve));
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Could not bind server');
  let registered = false;
  for (let i = 0; i < 5 && !registered; i++) {
    try {
      const response = await fetch(env.replace(/\/$/, '') + '/register', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }), signal: AbortSignal.timeout(5000) });
      if (response.ok) { const data = await response.json() as Obj; registered = data.ok === true; }
    } catch {}
    if (!registered) await delay(250);
  }
  if (!registered) { server.close(); throw new Error('Registration failed'); }
  const stop = () => { server.close(); process.exit(0); };
  process.once('SIGTERM', stop);
  process.once('SIGINT', stop);
  const poll = async () => {
    try {
      const response = await fetch(env.replace(/\/$/, '') + '/work', { signal: AbortSignal.timeout(5000) });
      if (response.ok && ((await response.json()) as Obj).done === true) { server.close(); process.exit(0); }
    } catch {}
    setTimeout(() => { void poll(); }, 100);
  };
  void poll();
}

main().catch(error => { process.stderr.write(String(error) + '\n'); process.exitCode = 1; });
