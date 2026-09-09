import { Type } from "typebox";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamAnthropic } from "../src/api/anthropic-messages.ts";
import { getModel } from "../src/compat.ts";
import type { AssistantMessage, Context, Model, Tool } from "../src/types.ts";

interface CapturedTool {
	name: string;
	defer_loading?: boolean;
}

interface CapturedParams {
	tools?: CapturedTool[];
	messages: Array<{ content: string | Array<{ type: string; content?: unknown }> }>;
}

const mockState = vi.hoisted(() => ({
	createParams: [] as Record<string, unknown>[],
	constructorOpts: [] as Record<string, unknown>[],
	failNextWith: undefined as Error | undefined,
}));

vi.mock("@anthropic-ai/sdk", () => {
	function createSseResponse(): Response {
		const body = [
			`event: message_start\ndata: ${JSON.stringify({
				type: "message_start",
				message: { id: "msg_test", usage: { input_tokens: 10, output_tokens: 0 } },
			})}\n`,
			`event: message_delta\ndata: ${JSON.stringify({
				type: "message_delta",
				delta: { stop_reason: "end_turn" },
				usage: { output_tokens: 5 },
			})}\n`,
			`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n`,
		].join("\n");

		return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
	}

	class FakeAnthropic {
		constructor(opts: Record<string, unknown>) {
			mockState.constructorOpts.push(opts);
		}
		messages = {
			create: (params: Record<string, unknown>) => {
				mockState.createParams.push(params);
				const failure = mockState.failNextWith;
				mockState.failNextWith = undefined;
				return {
					asResponse: async () => {
						if (failure) throw failure;
						return createSseResponse();
					},
				};
			},
		};
	}

	return { default: FakeAnthropic };
});

/** Shaped like the Anthropic SDK's `BadRequestError`, which carries `status` and `headers`. */
function badRequest(message: string): Error {
	const error = new Error(`400 ${message}`) as Error & { status: number; headers: Headers };
	error.status = 400;
	error.headers = new Headers();
	return error;
}

function makeTool(name: string): Tool {
	return { name, description: `The ${name} tool`, parameters: Type.Object({ value: Type.String() }) };
}

function makeContext(): Context {
	return {
		messages: [
			{ role: "user", content: "Hello", timestamp: 1 },
			{
				role: "assistant",
				content: [{ type: "toolCall", id: "call_1", name: "base_tool", arguments: {} }],
				api: "anthropic-messages",
				provider: "github-copilot",
				model: "claude-opus-5",
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
			},
			{
				role: "toolResult",
				toolCallId: "call_1",
				toolName: "base_tool",
				content: [{ type: "text", text: "done" }],
				addedToolNames: ["late_tool"],
				isError: false,
				timestamp: 3,
			},
			{ role: "user", content: "Continue", timestamp: 4 },
		],
		tools: [makeTool("base_tool"), makeTool("late_tool")],
	};
}

/** Each test needs its own endpoint key, because the downgrade is process-lifetime. */
function copilotModel(baseUrl: string): Model<"anthropic-messages"> {
	return { ...getModel("github-copilot", "claude-opus-5"), baseUrl };
}

async function run(model: Model<"anthropic-messages">, context: Context): Promise<AssistantMessage> {
	const s = streamAnthropic(model, context, { apiKey: "tid_copilot_session_test_token" });
	for await (const _event of s) {
		// Drain; the final message is read from result().
	}
	return await s.result();
}

/**
 * Client tool entries only. The server-side `tool_search_tool_bm25` entry is asserted
 * separately by `capturedSearchTool`, so the deferral assertions stay about the tools
 * mcpi actually registers.
 */
function capturedTools(index: number): CapturedTool[] {
	return ((mockState.createParams[index] as unknown as CapturedParams).tools ?? [])
		.filter((tool) => !tool.name.startsWith("tool_search_tool"))
		.map((tool) => ({
			name: tool.name,
			...(tool.defer_loading !== undefined ? { defer_loading: tool.defer_loading } : {}),
		}));
}

/** The server-side search entry sent with this request, if any. */
function capturedSearchTool(index: number): { name: string; type?: string } | undefined {
	const tools = (mockState.createParams[index] as unknown as CapturedParams).tools ?? [];
	const search = tools.find((tool) => tool.name.startsWith("tool_search_tool"));
	return search ? { name: search.name, type: (search as { type?: string }).type } : undefined;
}

describe("Anthropic deferred tools over the wire", () => {
	beforeEach(() => {
		mockState.createParams = [];
		mockState.constructorOpts = [];
		mockState.failNextWith = undefined;
	});

	it("sends defer_loading and a tool_reference to the Copilot gateway", async () => {
		const message = await run(copilotModel("https://api.individual.githubcopilot.com/probe-1"), makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
		expect(message.stopReason).toBe("stop");
		expect(message.diagnostics ?? []).toEqual([]);
	});

	// Without this entry on the wire, a deferred schema can only ever come back through a
	// tool result, so a tool no skill pushes would stay hidden for the whole session.
	it("sends the server-side search tool alongside deferred tools", async () => {
		await run(copilotModel("https://api.individual.githubcopilot.com/probe-search"), makeContext());

		expect(capturedSearchTool(0)).toEqual({
			name: "tool_search_tool_bm25",
			type: "tool_search_tool_bm25_20251119",
		});
	});

	it("sends no search tool when nothing is deferred", async () => {
		const context = makeContext();
		context.tools = [makeTool("base_tool")];
		await run(copilotModel("https://api.individual.githubcopilot.com/probe-nosearch"), context);

		expect(capturedSearchTool(0)).toBeUndefined();
	});

	// A rejection of the search tool is the same class of capability gap as a rejection of
	// `defer_loading`, so it downgrades once rather than failing the turn outright.
	it("downgrades once when the endpoint rejects the search tool", async () => {
		const model = copilotModel("https://proxy.invalid/rejects-search");
		mockState.failNextWith = badRequest(
			"Input tag 'tool_search_tool_bm25_20251119' found using 'type' does not match any of the expected tags",
		);

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(2);
		expect(capturedSearchTool(0)).toBeDefined();
		expect(capturedSearchTool(1)).toBeUndefined();
		expect(capturedTools(1)).toEqual([{ name: "base_tool" }, { name: "late_tool" }]);
		expect(message.stopReason).toBe("stop");
		expect(message.diagnostics?.[0]).toMatchObject({
			type: "deferred_tools_rejected",
			error: {
				message:
					"400 Input tag 'tool_search_tool_bm25_20251119' found using 'type' does not match any of the expected tags",
			},
		});
	});

	it("adds no anthropic-beta value on account of deferred tools", async () => {
		await run(copilotModel("https://api.individual.githubcopilot.com/probe-2"), makeContext());

		const headers = mockState.constructorOpts[0]?.defaultHeaders as Record<string, string> | undefined;
		const beta = headers?.["anthropic-beta"] ?? "";
		expect(beta).not.toContain("advanced-tool-use");
		expect(beta).not.toContain("tool-search");
		expect(beta).not.toContain("defer");
	});

	it("downgrades once and retries when the endpoint rejects defer_loading", async () => {
		const model = copilotModel("https://proxy.invalid/rejects-defer");
		mockState.failNextWith = badRequest("Extra inputs are not permitted: tools.1.defer_loading");

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(2);
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
		expect(capturedTools(1)).toEqual([{ name: "base_tool" }, { name: "late_tool" }]);
		expect(message.stopReason).toBe("stop");
		expect(message.diagnostics).toMatchObject([
			{
				type: "deferred_tools_rejected",
				error: { message: "400 Extra inputs are not permitted: tools.1.defer_loading" },
				details: { provider: "github-copilot", model: "claude-opus-5", deferredTools: ["late_tool"] },
			},
			{ type: "deferred_tools_unsupported", details: { deferredCandidates: ["late_tool"] } },
		]);
	});

	it("keeps the downgrade for the rest of the process without retrying again", async () => {
		const model = copilotModel("https://proxy.invalid/rejects-reference");
		mockState.failNextWith = badRequest(
			"Input tag 'tool_reference' found using 'type' does not match any of the expected tags",
		);
		await run(model, makeContext());
		expect(mockState.createParams).toHaveLength(2);

		mockState.createParams = [];
		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool" }]);
		expect(message.diagnostics).toMatchObject([{ type: "deferred_tools_unsupported" }]);
	});

	// An unresolvable reference proves the endpoint implements the protocol, so it is a
	// client activation or naming bug. Downgrading would hide it behind a silent
	// capability loss and keep sending full tool lists for the rest of the process.
	it("propagates an unknown tool_reference name without downgrading", async () => {
		const model = copilotModel("https://api.individual.githubcopilot.com/probe-names");
		mockState.failNextWith = badRequest("Tool reference 'late_tool' not found in available tools");

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toBe("400 Tool reference 'late_tool' not found in available tools");
		expect(message.diagnostics ?? []).toEqual([]);

		mockState.createParams = [];
		await run(model, makeContext());
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
	});

	it("propagates an underscore-spelled tool_reference resolution failure without downgrading", async () => {
		const model = copilotModel("https://api.individual.githubcopilot.com/probe-names-2");
		mockState.failNextWith = badRequest("tool_reference 'late_tool' does not exist in tools");

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(message.diagnostics ?? []).toEqual([]);

		mockState.createParams = [];
		await run(model, makeContext());
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
	});

	it("leaves other endpoints deferring after one endpoint rejects", async () => {
		mockState.failNextWith = badRequest("Extra inputs are not permitted: tools.1.defer_loading");
		await run(copilotModel("https://proxy.invalid/isolated-failure"), makeContext());

		mockState.createParams = [];
		await run(copilotModel("https://api.individual.githubcopilot.com/probe-3"), makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
	});

	it("propagates an unrelated 400 without retrying or downgrading", async () => {
		const model = copilotModel("https://api.individual.githubcopilot.com/probe-4");
		mockState.failNextWith = badRequest("model_not_supported: claude-opus-5");

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toBe("400 model_not_supported: claude-opus-5");
		expect(message.diagnostics ?? []).toEqual([]);

		mockState.createParams = [];
		await run(model, makeContext());
		expect(capturedTools(0)).toEqual([{ name: "base_tool" }, { name: "late_tool", defer_loading: true }]);
	});

	it("does not downgrade a model that was not deferring", async () => {
		const model: Model<"anthropic-messages"> = {
			...getModel("github-copilot", "claude-sonnet-4.6"),
			baseUrl: "https://api.individual.githubcopilot.com/probe-5",
		};
		mockState.failNextWith = badRequest("Extra inputs are not permitted: tools.1.defer_loading");

		const message = await run(model, makeContext());

		expect(mockState.createParams).toHaveLength(1);
		expect(message.stopReason).toBe("error");
		expect(message.errorMessage).toBe("400 Extra inputs are not permitted: tools.1.defer_loading");
	});
});
