import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { convertMessages } from "../src/api/openai-completions.ts";
import { getModel, streamSimple } from "../src/compat.ts";
import type { Api, AssistantMessage, Context, Model, Tool, ToolResultMessage, UserMessage } from "../src/types.ts";
import { estimateContextTokens } from "../src/utils/estimate.ts";

interface AnthropicToolPayload {
	name: string;
	description?: string;
	defer_loading?: boolean;
	cache_control?: { type: string; ttl?: string };
}

interface AnthropicContentBlock {
	type: string;
	text?: string;
	tool_use_id?: string;
	content?: string | Array<{ type: string; tool_name?: string }>;
	source?: {
		type: string;
		media_type: string;
		data: string;
	};
}

interface AnthropicPayload {
	tools?: AnthropicToolPayload[];
	messages: Array<{
		content: string | AnthropicContentBlock[];
	}>;
}

interface OpenAIToolSearchCall {
	type: "tool_search_call";
	call_id?: string | null;
	execution?: string;
	status?: string | null;
}

interface OpenAIToolSearchOutput {
	type: "tool_search_output";
	call_id?: string | null;
	execution?: string;
	status?: string | null;
	tools: Array<{ type: string; name: string; defer_loading?: boolean }>;
}

interface OpenAIAdditionalTools {
	type: "additional_tools";
	role: "developer";
	tools: Array<{ type: string; name: string; defer_loading?: boolean }>;
}

interface OpenAIPayload {
	tools?: Array<{ name?: string; function?: { name: string } }>;
	input?: Array<
		OpenAIAdditionalTools | OpenAIToolSearchCall | OpenAIToolSearchOutput | { type?: string; name?: string }
	>;
}

interface KimiTool {
	type: "function";
	function: {
		name: string;
		description?: string;
		parameters?: Record<string, unknown>;
	};
}

interface KimiMessage {
	role: string;
	content?: unknown;
	tools?: KimiTool[];
}

interface KimiPayload {
	tools?: KimiTool[];
	messages: KimiMessage[];
}

class PayloadCaptured extends Error {}

function makeTool(name: string): Tool {
	return {
		name,
		description: `The ${name} tool`,
		parameters: Type.Object({ value: Type.String() }),
	};
}

function makeUserMessage(timestamp: number): UserMessage {
	return { role: "user", content: "Hello", timestamp };
}

function makeAssistantToolCall(): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "toolCall", id: "call_1", name: "base_tool", arguments: {} }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-opus-4-6",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "toolUse",
		timestamp: 2,
	};
}

function makeToolResult(addedToolNames: string[]): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: "call_1",
		toolName: "base_tool",
		content: [{ type: "text", text: "done" }],
		addedToolNames,
		isError: false,
		timestamp: 3,
	};
}

function makeContext(tools: Tool[], addedToolNames = ["late_tool"]): Context {
	return {
		messages: [makeUserMessage(1), makeAssistantToolCall(), makeToolResult(addedToolNames), makeUserMessage(4)],
		tools,
	};
}

function makeKimiModel(deferredToolsMode?: "kimi"): Model<"openai-completions"> {
	return {
		id: "deferred-tools-model",
		name: "Deferred Tools Model",
		api: "openai-completions",
		provider: "moonshotai",
		baseUrl: "http://127.0.0.1:9/v1",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
		compat: deferredToolsMode ? { deferredToolsMode } : undefined,
	};
}

async function capturePayload<T>(model: Model<Api>, context: Context, apiKey = "fake-key"): Promise<T> {
	let captured: T | undefined;
	const stream = streamSimple({ ...model, baseUrl: "http://127.0.0.1:9" }, context, {
		apiKey,
		onPayload: (payload) => {
			captured = payload as T;
			throw new PayloadCaptured();
		},
	});
	await stream.result();
	if (!captured) throw new Error("Expected payload capture");
	return captured;
}

async function captureAssistantMessage(model: Model<Api>, context: Context): Promise<AssistantMessage> {
	const stream = streamSimple({ ...model, baseUrl: "http://127.0.0.1:9" }, context, {
		apiKey: "fake-key",
		onPayload: () => {
			throw new PayloadCaptured();
		},
	});
	return await stream.result();
}

function findAnthropicToolResultContent(payload: AnthropicPayload): AnthropicContentBlock[] {
	for (const message of payload.messages) {
		if (typeof message.content !== "string" && message.content.some((block) => block.type === "tool_result")) {
			return message.content;
		}
	}
	throw new Error("No tool result in payload");
}

function findAnthropicToolResult(payload: AnthropicPayload): AnthropicContentBlock {
	const result = findAnthropicToolResultContent(payload).find((block) => block.type === "tool_result");
	if (!result) throw new Error("No tool result in payload");
	return result;
}

function openAIToolNames(payload: OpenAIPayload): string[] {
	return (payload.tools ?? []).map((tool) => tool.name ?? tool.function?.name ?? "");
}

function makeCodexToken(): string {
	return `header.${btoa(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "account" } }))}.signature`;
}

/**
 * Appended by the Anthropic provider whenever a request carries deferred tools and the
 * model is verified to implement server-side search. It is what makes a deferred tool
 * reachable when no tool result ever names it.
 */
const TOOL_SEARCH_ENTRY = { name: "tool_search_tool_bm25", type: "tool_search_tool_bm25_20251119" };

function toolNames(payload: AnthropicPayload): string[] {
	return (payload.tools ?? []).map((tool) => tool.name);
}

describe("deferred tools", () => {
	it("loads an Anthropic tool at its tool-result marker", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("loads a Claude Opus 5 tool at its tool-result marker", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("keeps a Claude Opus 5 tool immediate when it was used before its marker", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [{ type: "toolCall", id: "call_1", name: "late_tool", arguments: {} }];
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-5"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("defers on the GitHub Copilot Opus 5 variant", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	// Registration deferral, unlike a transcript marker, has to work on the very first
	// request, when there is no tool result to anchor a load point to.
	it("hides a registration-deferred schema on turn zero on GitHub Copilot Opus 5", async () => {
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		// `defer_loading` is what keeps the schema out of the model's context: the
		// definition still travels on the wire, and the gateway withholds it until a
		// `tool_reference` loads it. Without this flag on turn zero the schema is in
		// context from the first request and the saving never happens.
		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(payload.tools?.find((tool) => tool.name === "base_tool")?.defer_loading).toBeUndefined();
		// No load point exists yet, so nothing is referenced back.
		expect(JSON.stringify(payload.messages)).not.toContain("tool_reference");
	});

	it("loads a turn-zero registration-deferred schema at the addedToolNames marker on GitHub Copilot", async () => {
		const context = makeContext([makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }]);
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("expands a turn-zero registration-deferred schema on an unsupported Copilot model and reports it", async () => {
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const model = getModel("github-copilot", "claude-sonnet-4.6");
		const payload = await capturePayload<AnthropicPayload>(model, context);
		const message = await captureAssistantMessage(model, context);

		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		expect(message.diagnostics).toMatchObject([
			{
				type: "deferred_tools_unsupported",
				details: { provider: "github-copilot", deferredCandidates: ["late_tool"] },
			},
		]);
	});

	// The cacheable prefix is the tools array plus the system prompt. Activating a
	// registration-deferred tool must not reorder or re-describe the immediate entries,
	// or every cached prefix built before activation is invalidated.
	it("keeps the immediate tool prefix stable when a registration-deferred tool activates", async () => {
		const tools = [makeTool("base_tool"), makeTool("second_tool"), { ...makeTool("late_tool"), deferred: true }];
		const model = getModel("github-copilot", "claude-opus-5");
		const before = await capturePayload<AnthropicPayload>(model, { messages: [makeUserMessage(1)], tools });
		const after = await capturePayload<AnthropicPayload>(model, makeContext(tools));

		expect(after.tools?.map((tool) => tool.name)).toEqual(before.tools?.map((tool) => tool.name));
		expect(after.tools?.map((tool) => tool.defer_loading)).toEqual(before.tools?.map((tool) => tool.defer_loading));
		const immediateBefore = before.tools?.filter((tool) => !tool.defer_loading);
		expect(after.tools?.filter((tool) => !tool.defer_loading)).toEqual(immediateBefore);
	});

	// Every id here was verified against the Copilot gateway with a live probe.
	// Dotted minor versions are the reason this is an explicit allowlist: the
	// version parser used for direct Anthropic never matched them.
	it.each(["claude-opus-5", "claude-sonnet-5", "claude-opus-4.8", "claude-opus-4.7", "claude-haiku-4.5"])(
		"defers on the verified GitHub Copilot model %s",
		async (modelId) => {
			const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
			const model = getModel("github-copilot", modelId as "claude-opus-5");
			expect(model.api).toBe("anthropic-messages");
			expect(model.compat?.supportsToolReferences).toBe(true);

			const payload = await capturePayload<AnthropicPayload>(model, context);

			expect(payload.tools).toMatchObject([
				{ name: "base_tool" },
				{ name: "late_tool", defer_loading: true },
				TOOL_SEARCH_ENTRY,
			]);
			expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
		},
	);

	// The Copilot gateway rejects claude-sonnet-4.6 outright, so it is deliberately
	// absent from the verified allowlist. This pins that Claude version alone never
	// implies deferral on a gateway.
	it("does not defer on a GitHub Copilot Claude model outside the verified allowlist", async () => {
		const model = getModel("github-copilot", "claude-sonnet-4.6");
		expect(model.compat?.supportsToolReferences).toBeUndefined();

		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(false);
	});

	// An unprobed Copilot Claude id must fall back rather than inherit deferral from
	// a sibling. Mutating the allowlist to a prefix or provider-wide test would break
	// this, which is the point.
	it("does not defer on an unknown GitHub Copilot Claude id", async () => {
		const model: Model<"anthropic-messages"> = {
			...getModel("github-copilot", "claude-opus-5"),
			id: "claude-opus-6",
			compat: { forceAdaptiveThinking: true, supportsTemperature: false },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("keeps Copilot deferral off when a model override disables it", async () => {
		const model: Model<"anthropic-messages"> = {
			...getModel("github-copilot", "claude-opus-5"),
			compat: { ...getModel("github-copilot", "claude-opus-5").compat, supportsToolReferences: false },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("keeps a Copilot deferred tool immediate once the model has called it", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [{ type: "toolCall", id: "call_1", name: "late_tool", arguments: {} }];
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	// The all-deferred floor exists so the model is never left with no schema and no way to
	// find one. Where the search tool is sent that premise does not hold, so the floor stands
	// down and the sole tool stays deferred -- its marker still loads it at the tool result.
	it("defers the only Copilot tool because search can still reach it", async () => {
		const context = makeContext([makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([{ name: "late_tool", defer_loading: true }, TOOL_SEARCH_ENTRY]);
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(true);
	});

	// The gateway 400s on a tool_reference mixed with other content in one
	// tool_result, so sibling output must be displaced after the tool_result block.
	it("never mixes a Copilot tool_reference with other tool-result content", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const firstResult = context.messages[2] as ToolResultMessage;
		firstResult.content = [
			{ type: "text", text: "work completed" },
			{ type: "image", mimeType: "image/png", data: "aW1hZ2U=" },
		];

		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(findAnthropicToolResultContent(payload)).toMatchObject([
			{
				type: "tool_result",
				tool_use_id: "call_1",
				content: [{ type: "tool_reference", tool_name: "late_tool" }],
			},
			{ type: "text", text: "work completed" },
			{ type: "image", source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" } },
		]);
		const toolResult = findAnthropicToolResult(payload);
		expect(Array.isArray(toolResult.content) && toolResult.content.every((b) => b.type === "tool_reference")).toBe(
			true,
		);
	});

	// A marker is a permanent transcript fact: it must still defer many turns later,
	// and the reference must stay at its original index so the cached prefix and the
	// tool array are byte-stable across turns.
	it("keeps a Copilot marker deferred and referenced once across later turns", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const firstPayload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		const laterContext: Context = {
			...context,
			messages: [
				...context.messages,
				{
					...makeAssistantToolCall(),
					content: [{ type: "toolCall", id: "call_2", name: "base_tool", arguments: {} }],
					provider: "github-copilot",
					model: "claude-opus-5",
					timestamp: 5,
				},
				{ ...makeToolResult([]), toolCallId: "call_2", timestamp: 6 },
				makeUserMessage(7),
			],
		};
		const laterPayload = await capturePayload<AnthropicPayload>(
			getModel("github-copilot", "claude-opus-5"),
			laterContext,
		);

		expect(laterPayload.tools).toEqual(firstPayload.tools);
		const references = laterPayload.messages.flatMap((message) =>
			typeof message.content === "string"
				? []
				: message.content.flatMap((block) =>
						Array.isArray(block.content) ? block.content.filter((inner) => inner.type === "tool_reference") : [],
					),
		);
		expect(references).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	// Duplicate markers for one tool must collapse to a single reference; the gateway
	// treats a repeated definition as a protocol error.
	it("emits a Copilot tool_reference once when several results mark the same tool", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [
			{ type: "toolCall", id: "call_1", name: "base_tool", arguments: {} },
			{ type: "toolCall", id: "call_2", name: "base_tool", arguments: {} },
		];
		context.messages.splice(3, 0, { ...makeToolResult(["late_tool"]), toolCallId: "call_2" });

		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		const references = findAnthropicToolResultContent(payload).flatMap((block) =>
			Array.isArray(block.content) ? block.content.filter((inner) => inner.type === "tool_reference") : [],
		);
		expect(references).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	// Deferred definitions must sit after the cache breakpoint so the cached prefix
	// is not invalidated when a tool is added mid-session.
	it("puts the Copilot cache breakpoint on the last immediate tool", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("other_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		// The search tool is appended after the deferred entries, so it sits outside the
		// prefix the breakpoint closes and cannot shift the cached region.
		expect(payload.tools?.map((tool) => tool.name)).toEqual([
			"base_tool",
			"other_tool",
			"late_tool",
			"tool_search_tool_bm25",
		]);
		expect(payload.tools?.at(-1)).toMatchObject(TOOL_SEARCH_ENTRY);
		expect(payload.tools?.at(-1)?.cache_control).toBeUndefined();
		expect(payload.tools?.at(-2)).toMatchObject({ name: "late_tool", defer_loading: true });
		expect(payload.tools?.at(-2)?.cache_control).toBeUndefined();
		expect(payload.tools?.[1]).toMatchObject({ name: "other_tool", cache_control: { type: "ephemeral" } });
		expect(payload.tools?.[0]?.cache_control).toBeUndefined();
	});

	it("loads a tool introduced by direct Anthropic history after switching to Copilot", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.provider = "anthropic";
		assistant.model = "claude-opus-5";

		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("preserves tool output as sibling content after emitting references", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [
			{ type: "toolCall", id: "call_1", name: "base_tool", arguments: {} },
			{ type: "toolCall", id: "call_2", name: "base_tool", arguments: {} },
		];
		const firstResult = context.messages[2] as ToolResultMessage;
		firstResult.content = [
			{ type: "text", text: "work completed" },
			{ type: "image", mimeType: "image/png", data: "aW1hZ2U=" },
		];
		context.messages.splice(3, 0, {
			...makeToolResult([]),
			toolCallId: "call_2",
			content: [{ type: "text", text: "second result" }],
		});

		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(findAnthropicToolResultContent(payload)).toMatchObject([
			{
				type: "tool_result",
				tool_use_id: "call_1",
				content: [{ type: "tool_reference", tool_name: "late_tool" }],
			},
			{ type: "tool_result", tool_use_id: "call_2", content: "second result" },
			{ type: "text", text: "work completed" },
			{
				type: "image",
				source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" },
			},
		]);
	});

	it("loads a tool introduced by OpenAI history after switching to Anthropic", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.api = "openai-responses";
		assistant.provider = "openai";
		assistant.model = "gpt-5.4";

		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-8"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("loads a tool introduced by OpenAI history after switching to Anthropic Opus 5", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.api = "openai-responses";
		assistant.provider = "openai";
		assistant.model = "gpt-5.4";

		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	// The transcript stays on anthropic-messages across this switch; only the provider changes,
	// so this pins that the marker survives a provider hop within one API family.
	it("loads a tool introduced by Copilot Opus 5 history after switching to direct Anthropic", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.provider = "github-copilot";
		assistant.model = "claude-opus-5";

		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-5"), context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
	});

	it("does not resurrect a marked tool missing from Context.tools", async () => {
		const context = makeContext([makeTool("base_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool"]);
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(false);
	});

	it("keeps a tool immediate when it was used before its marker", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [{ type: "toolCall", id: "call_1", name: "late_tool", arguments: {} }];
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("normalizes OAuth names before checking prior tool usage", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("read")], ["read"]);
		const assistant = context.messages[1] as AssistantMessage;
		assistant.content = [{ type: "toolCall", id: "call_1", name: "Read", arguments: {} }];
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "Read"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(false);
	});

	it("matches OAuth-canonicalized markers to active tools", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("read")], ["Read"]);
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "Read", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		const content = findAnthropicToolResult(payload).content;
		expect(
			Array.isArray(content) &&
				content.some((block) => block.type === "tool_reference" && block.tool_name === "Read"),
		).toBe(true);
	});

	it("deduplicates active tools after OAuth canonicalization", async () => {
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("read"), { ...makeTool("Read"), description: "Canonical definition" }],
		};
		const payload = await capturePayload<AnthropicPayload>(
			getModel("anthropic", "claude-opus-4-6"),
			context,
			"sk-ant-oat-fake",
		);

		expect(payload.tools).toMatchObject([{ name: "Read", description: "Canonical definition" }]);
	});

	it("uses the normal tool list when Anthropic tool references are unsupported", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const models: Model<"anthropic-messages">[] = [
			getModel("anthropic", "claude-haiku-4-5"),
			{ ...getModel("anthropic", "claude-opus-4-6"), id: "claude-sonnet-4-20250514" },
		];

		for (const model of models) {
			const payload = await capturePayload<AnthropicPayload>(model, context);
			expect(payload.tools?.map((tool) => tool.name)).toEqual(["base_tool", "late_tool"]);
			expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		}
	});

	it("defers the only Anthropic tool because search can still reach it", async () => {
		const context = makeContext([makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("anthropic", "claude-opus-4-6"), context);

		expect(payload.tools).toMatchObject([{ name: "late_tool", defer_loading: true }, TOOL_SEARCH_ENTRY]);
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(true);
	});

	// Without search the original reasoning applies again: nothing would carry a schema.
	it("keeps the only tool immediate when search is switched off", async () => {
		const base = getModel("anthropic", "claude-opus-4-6");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };
		const payload = await capturePayload<AnthropicPayload>(model, makeContext([makeTool("late_tool")]));

		expect(payload.tools).toMatchObject([{ name: "late_tool" }]);
		expect(payload.tools?.[0]?.defer_loading).toBeUndefined();
		const content = findAnthropicToolResult(payload).content;
		expect(Array.isArray(content) && content.some((block) => block.type === "tool_reference")).toBe(false);
	});

	it("supports explicit Anthropic compatibility overrides", async () => {
		const model: Model<"anthropic-messages"> = {
			...getModel("anthropic", "claude-opus-4-6"),
			provider: "anthropic-proxy",
			compat: { supportsToolReferences: true },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools?.find((tool) => tool.name === "late_tool")?.defer_loading).toBe(true);
	});

	// The version fallback used to accept only dashed minor versions, so a dotted id
	// silently fell back even when the family was new enough.
	it.each(["claude-opus-4.8", "claude-opus-4.7", "claude-fable-5.1", "claude-sonnet-4.5"])(
		"parses the dotted minor version in %s",
		async (modelId) => {
			const model: Model<"anthropic-messages"> = {
				...getModel("anthropic", "claude-opus-4-6"),
				id: modelId,
				compat: undefined,
			};
			const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
			const payload = await capturePayload<AnthropicPayload>(model, context);

			expect(payload.tools).toMatchObject([
				{ name: "base_tool" },
				{ name: "late_tool", defer_loading: true },
				TOOL_SEARCH_ENTRY,
			]);
		},
	);

	it.each(["claude-opus-4.1", "claude-sonnet-4.0", "claude-sonnet-4-20250514", "claude-haiku-4.5"])(
		"does not defer on %s",
		async (modelId) => {
			const model: Model<"anthropic-messages"> = {
				...getModel("anthropic", "claude-opus-4-6"),
				id: modelId,
				compat: undefined,
			};
			const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
			const payload = await capturePayload<AnthropicPayload>(model, context);

			expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		},
	);

	it("records a diagnostic when a marked tool is expanded because deferral is unsupported", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const message = await captureAssistantMessage(getModel("github-copilot", "claude-sonnet-4.6"), context);

		expect(message.diagnostics).toMatchObject([
			{
				type: "deferred_tools_unsupported",
				details: {
					provider: "github-copilot",
					model: "claude-sonnet-4.6",
					deferredCandidates: ["late_tool"],
				},
			},
		]);
	});

	it("records no deferral diagnostic when the model defers the marked tool", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const message = await captureAssistantMessage(getModel("github-copilot", "claude-opus-5"), context);

		expect(message.diagnostics ?? []).toEqual([]);
	});

	it("records no deferral diagnostic when the transcript marks nothing", async () => {
		const context: Context = { messages: [makeUserMessage(1)], tools: [makeTool("base_tool")] };
		const message = await captureAssistantMessage(getModel("github-copilot", "claude-sonnet-4.6"), context);

		expect(message.diagnostics ?? []).toEqual([]);
	});

	it("serializes Kimi deferred tools as system tool definitions", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<KimiPayload>(makeKimiModel("kimi"), context);

		expect(payload.tools?.map((tool) => tool.function.name)).toEqual(["base_tool"]);
		const toolResultIndex = payload.messages.findIndex((message) => message.role === "tool");
		const systemToolIndex = payload.messages.findIndex((message) => message.tools !== undefined);
		expect(toolResultIndex).toBeGreaterThanOrEqual(0);
		expect(systemToolIndex).toBeGreaterThan(toolResultIndex);
		expect(payload.messages[systemToolIndex]?.tools?.map((tool) => tool.function.name)).toEqual(["late_tool"]);
	});

	it("emits Kimi deferred schemas after all tool results in a batch", () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool"), makeTool("later_tool")]);
		context.messages.splice(3, 0, {
			...makeToolResult(["later_tool"]),
			toolCallId: "call_2",
		});

		const messages = convertMessages(makeKimiModel("kimi"), context, {
			supportsStore: false,
			supportsDeveloperRole: false,
			supportsReasoningEffort: false,
			supportsUsageInStreaming: true,
			supportsFinishReason: true,
			maxTokensField: "max_tokens",
			requiresToolResultName: false,
			requiresAssistantAfterToolResult: false,
			requiresThinkingAsText: false,
			requiresReasoningContentOnAssistantMessages: false,
			thinkingFormat: "openai",
			openRouterRouting: {},
			vercelGatewayRouting: {},
			chatTemplateKwargs: {},
			chatTemplateArgs: {},
			zaiToolStream: false,
			supportsStrictMode: false,
			supportsOpenAIGrammarTools: false,
			cacheControlFormat: undefined,
			sendSessionAffinityHeaders: false,
			deferredToolsMode: "kimi",
			sessionAffinityFormat: "openai",
			supportsLongCacheRetention: false,
		});

		expect(messages.map((message) => message.role)).toEqual(["user", "assistant", "tool", "tool", "system", "user"]);
		expect((messages[4] as { tools?: KimiTool[] }).tools?.map((tool) => tool.function.name)).toEqual([
			"late_tool",
			"later_tool",
		]);
	});

	it("leaves OpenAI Completions tools unchanged without Kimi mode", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<KimiPayload>(makeKimiModel(), context);

		expect(payload.tools?.map((tool) => tool.function.name)).toEqual(["base_tool", "late_tool"]);
		expect(payload.messages.some((message) => message.tools !== undefined)).toBe(false);
	});

	it("loads an OpenAI Responses tool through additional_tools", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<OpenAIPayload>(getModel("openai", "gpt-5.4"), context);
		const additionalTools = payload.input?.find(
			(item): item is OpenAIAdditionalTools => item.type === "additional_tools",
		);

		expect(openAIToolNames(payload)).toEqual(["base_tool"]);
		expect(additionalTools).toMatchObject({ role: "developer" });
		expect(additionalTools?.tools).toMatchObject([{ type: "function", name: "late_tool" }]);
		expect(additionalTools?.tools.every((tool) => tool.defer_loading === undefined)).toBe(true);
		expect(payload.input?.some((item) => item.type === "tool_search_call")).toBe(false);
		expect(payload.input?.some((item) => item.type === "tool_search_output")).toBe(false);
	});

	it("preserves an additional_tools marker after the loaded tool is used", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const lateCall: AssistantMessage = {
			...makeAssistantToolCall(),
			content: [{ type: "toolCall", id: "call_late|fc_late", name: "late_tool", arguments: {} }],
			api: "openai-responses",
			provider: "openai",
			model: "gpt-5.4",
		};
		context.messages.splice(3, 0, lateCall, {
			...makeToolResult(["late_tool"]),
			toolCallId: "call_late|fc_late",
			toolName: "late_tool",
		});

		const payload = await capturePayload<OpenAIPayload>(getModel("openai", "gpt-5.4"), context);
		const additionalToolIndexes = (payload.input ?? []).flatMap((item, index) =>
			item.type === "additional_tools" ? [index] : [],
		);
		const lateCallIndex = (payload.input ?? []).findIndex(
			(item) => item.type === "function_call" && item.name === "late_tool",
		);

		expect(additionalToolIndexes).toHaveLength(1);
		expect(additionalToolIndexes[0]).toBeLessThan(lateCallIndex);
		expect(openAIToolNames(payload)).toEqual(["base_tool"]);
	});

	it("falls back to client tool search when additional_tools is unsupported", async () => {
		const model: Model<"openai-responses"> = {
			...getModel("openai", "gpt-5.4"),
			provider: "openai-proxy",
			compat: { supportsAdditionalTools: false, supportsToolSearch: true },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<OpenAIPayload>(model, context);
		const searchCall = payload.input?.find((item): item is OpenAIToolSearchCall => item.type === "tool_search_call");
		const searchOutput = payload.input?.find(
			(item): item is OpenAIToolSearchOutput => item.type === "tool_search_output",
		);

		expect(openAIToolNames(payload)).toEqual(["base_tool"]);
		expect(searchCall).toMatchObject({ execution: "client", status: "completed" });
		expect(searchOutput?.call_id).toBe(searchCall?.call_id);
		expect(searchOutput?.tools).toMatchObject([{ type: "function", name: "late_tool", defer_loading: true }]);
		expect(payload.input?.some((item) => item.type === "additional_tools")).toBe(false);
	});

	it.each(["gpt-5.2", "gpt-5.4-nano", "gpt-5.5-pro"] as const)(
		"uses the normal tool list for unsupported OpenAI model %s",
		async (modelId) => {
			const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
			const payload = await capturePayload<OpenAIPayload>(getModel("openai", modelId), context);

			expect(openAIToolNames(payload)).toEqual(["base_tool", "late_tool"]);
			expect(payload.input?.some((item) => item.type === "tool_search_output")).toBe(false);
		},
	);

	it("uses the normal tool list when OpenAI tool search is explicitly disabled", async () => {
		const model: Model<"openai-responses"> = {
			...getModel("openai", "gpt-5.4"),
			provider: "openai-proxy",
			compat: { supportsToolSearch: false },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<OpenAIPayload>(model, context);

		expect(openAIToolNames(payload)).toEqual(["base_tool", "late_tool"]);
		expect(payload.input?.some((item) => item.type === "tool_search_output")).toBe(false);
	});

	it("selects additional tools, tool search, or top-level tools for Codex models", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const additionalTools = await capturePayload<OpenAIPayload>(
			getModel("openai-codex", "gpt-5.6-sol"),
			context,
			makeCodexToken(),
		);
		const toolSearch = await capturePayload<OpenAIPayload>(
			getModel("openai-codex", "gpt-5.4"),
			context,
			makeCodexToken(),
		);
		const topLevel = await capturePayload<OpenAIPayload>(
			getModel("openai-codex", "gpt-5.3-codex-spark"),
			context,
			makeCodexToken(),
		);

		expect(openAIToolNames(additionalTools)).toEqual(["base_tool"]);
		expect(additionalTools.input?.some((item) => item.type === "additional_tools")).toBe(true);
		expect(additionalTools.input?.some((item) => item.type === "tool_search_output")).toBe(false);
		expect(openAIToolNames(toolSearch)).toEqual(["base_tool"]);
		expect(toolSearch.input?.some((item) => item.type === "tool_search_output")).toBe(true);
		expect(openAIToolNames(topLevel)).toEqual(["base_tool", "late_tool"]);
		expect(topLevel.input?.some((item) => item.type === "additional_tools")).toBe(false);
		expect(topLevel.input?.some((item) => item.type === "tool_search_output")).toBe(false);
	});

	it("leaves providers without deferred loading unchanged", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<OpenAIPayload>(getModel("groq", "llama-3.3-70b-versatile"), context);
		expect(openAIToolNames(payload)).toEqual(["base_tool", "late_tool"]);
	});

	it("counts definitions marked after the latest usage checkpoint", () => {
		const assistant: AssistantMessage = {
			...makeAssistantToolCall(),
			content: [{ type: "text", text: "done" }],
			usage: {
				input: 50,
				output: 50,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 100,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
		};
		const plain = estimateContextTokens({ messages: [assistant, makeUserMessage(4)], tools: [] });
		const lateTool = { ...makeTool("late_tool"), description: "x".repeat(4000) };
		const marked = estimateContextTokens({
			messages: [assistant, makeToolResult(["late_tool"])],
			tools: [lateTool],
		});

		expect(marked.tokens).toBeGreaterThan(plain.tokens + 500);
		expect(marked.trailingTokens).toBeGreaterThan(plain.trailingTokens + 500);
	});
});

/**
 * Skills push a tool into context by naming it in `ToolResultMessage.addedToolNames`.
 * A tool registered with `deferred: true` that no skill ever names has no such push, so
 * provider-native search is the only thing that can reach it. Without search, deferring it
 * would hide it for the whole session, which is worse than never deferring it at all.
 */
describe("provider-native tool search", () => {
	it("offers search alongside an unskilled registration-deferred tool on turn zero", async () => {
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		// The schema is withheld, and the model is told it can go looking for it.
		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
		// Nothing pushed the tool, so there is no reference anywhere in the transcript.
		expect(JSON.stringify(payload.messages)).not.toContain("tool_reference");
	});

	it("does not offer search when nothing is deferred", async () => {
		const context: Context = { messages: [makeUserMessage(1)], tools: [makeTool("base_tool")] };
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(toolNames(payload)).toEqual(["base_tool"]);
	});

	it("does not offer search when no tools are registered", async () => {
		const context: Context = { messages: [makeUserMessage(1)], tools: [] };
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		expect(payload.tools).toBeUndefined();
	});

	it("never defers the search tool itself", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);
		const search = payload.tools?.find((tool) => tool.name === "tool_search_tool_bm25");

		// A deferred search tool could only be loaded by searching for it.
		expect(search?.defer_loading).toBeUndefined();
	});

	// The safety net. Search is on wherever `tool_reference` resolves, but a gateway can
	// resolve references and still reject the search tool; `supportsToolSearch: false` says so.
	// A registration-deferred tool then has no way back, so its schema must be sent up front.
	it("expands an unskilled registration-deferred tool when the model cannot search", async () => {
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const base = getModel("anthropic", "claude-opus-5");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };

		const payload = await capturePayload<AnthropicPayload>(model, context);
		const message = await captureAssistantMessage(model, context);

		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
		expect(message.diagnostics).toMatchObject([
			{ type: "deferred_tools_unsupported", details: { deferredCandidates: ["late_tool"] } },
		]);
	});

	// Losing registration deferral must not cost the skill-driven path, which anchors its
	// load point to a tool result and therefore works with no search tool at all.
	it("still defers a skill-pushed tool when the model cannot search", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const base = getModel("anthropic", "claude-opus-5");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };
		const payload = await capturePayload<AnthropicPayload>(model, context);
		const message = await captureAssistantMessage(model, context);

		expect(payload.tools).toMatchObject([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
		expect(toolNames(payload)).not.toContain("tool_search_tool_bm25");
		expect(findAnthropicToolResult(payload).content).toEqual([{ type: "tool_reference", tool_name: "late_tool" }]);
		expect(message.diagnostics ?? []).toEqual([]);
	});

	// A tool registered as deferred but pushed by a skill keeps its anchor, so it stays
	// deferred even on a model that cannot search.
	// With search off there is no turn-zero route back to a registration-deferred schema, so
	// it is sent up front and reported. It stays up front even though a marker later names it:
	// moving an already-sent schema to a later load point would churn the cached prefix.
	it("sends a registration-deferred tool up front and reports it when search is off", async () => {
		const context = makeContext([makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }]);
		const base = getModel("anthropic", "claude-opus-5");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };
		const payload = await capturePayload<AnthropicPayload>(model, context);
		const message = await captureAssistantMessage(model, context);

		expect(payload.tools).toMatchObject([{ name: "base_tool" }, { name: "late_tool" }]);
		expect(payload.tools?.every((tool) => tool.defer_loading === undefined)).toBe(true);
		expect(message.diagnostics?.map((entry) => entry.type)).toEqual(["deferred_tools_unsupported"]);
	});

	// Marker-driven deferral needs no search tool: its load point is the tool result itself.
	it("still defers a marker-pushed tool when search is off", async () => {
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const base = getModel("anthropic", "claude-opus-5");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };
		const payload = await capturePayload<AnthropicPayload>(model, context);
		const message = await captureAssistantMessage(model, context);

		expect(payload.tools).toMatchObject([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
		expect(message.diagnostics ?? []).toEqual([]);
	});

	// The second half of the contract: once search loads a schema the model calls the tool
	// directly, and that call has to keep working on the next request.
	it("promotes a searched-and-called tool to immediate on the following request", async () => {
		const tools = [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }];
		// The model found `late_tool` through search and called it. No tool result ever
		// named it, so the only record that its schema was loaded is the call itself.
		const assistant = makeAssistantToolCall();
		assistant.content = [{ type: "toolCall", id: "call_1", name: "late_tool", arguments: {} }];
		assistant.provider = "github-copilot";
		assistant.model = "claude-opus-5";
		const result = makeToolResult([]);
		result.toolName = "late_tool";
		const context: Context = { messages: [makeUserMessage(1), assistant, result, makeUserMessage(4)], tools };
		const payload = await capturePayload<AnthropicPayload>(getModel("github-copilot", "claude-opus-5"), context);

		// The transcript holds a `tool_use` for it, so its schema must be present or the
		// replayed history is invalid. Nothing is left deferred, so no search tool is sent.
		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("honors a models.json override that disables search", async () => {
		const base = getModel("github-copilot", "claude-opus-5");
		const model = { ...base, compat: { ...base.compat, supportsToolSearch: false } };
		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const payload = await capturePayload<AnthropicPayload>(model, context);

		// Search off means registration deferral is unsafe, so the schema is sent up front.
		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	// Anthropic couples the two features: a deferred schema is "only loaded when returned via
	// tool_reference from tool search". So search follows reference support rather than needing
	// its own allowlist, and first-party Anthropic gets it without any override.
	it("offers search by default wherever tool references resolve", async () => {
		const model = getModel("anthropic", "claude-opus-5");
		expect(model.compat?.supportsToolSearch).toBeUndefined();

		const context: Context = {
			messages: [makeUserMessage(1)],
			tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
		};
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(payload.tools).toMatchObject([
			{ name: "base_tool" },
			{ name: "late_tool", defer_loading: true },
			TOOL_SEARCH_ENTRY,
		]);
	});

	// Search returns `tool_reference` blocks, so it is useless on an endpoint that cannot
	// resolve them. Offering it there would add a tool the model can only fail with.
	it("ignores enabled search when the model cannot resolve references", async () => {
		const base = getModel("github-copilot", "claude-opus-5");
		const model = {
			...base,
			compat: { ...base.compat, supportsToolReferences: false, supportsToolSearch: true },
		};
		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
		expect(payload.tools?.every((tool) => !tool.defer_loading)).toBe(true);
	});

	it("offers search on every verified GitHub Copilot model", async () => {
		for (const modelId of [
			"claude-opus-5",
			"claude-sonnet-5",
			"claude-opus-4.8",
			"claude-opus-4.7",
			"claude-haiku-4.5",
		]) {
			const model = getModel("github-copilot", modelId as "claude-opus-5");

			const context: Context = {
				messages: [makeUserMessage(1)],
				tools: [makeTool("base_tool"), { ...makeTool("late_tool"), deferred: true }],
			};
			const payload = await capturePayload<AnthropicPayload>(model, context);

			expect(payload.tools?.at(-1)).toMatchObject(TOOL_SEARCH_ENTRY);
			expect(payload.tools?.find((tool) => tool.name === "late_tool")?.defer_loading).toBe(true);
		}
	});

	it("does not offer search on a Copilot model outside the verified allowlist", async () => {
		const model = getModel("github-copilot", "claude-sonnet-4.6");
		expect(model.compat?.supportsToolSearch).toBeUndefined();

		const context = makeContext([makeTool("base_tool"), makeTool("late_tool")]);
		const payload = await capturePayload<AnthropicPayload>(model, context);

		expect(toolNames(payload)).toEqual(["base_tool", "late_tool"]);
	});
});
