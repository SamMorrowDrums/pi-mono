import { describe, expect, it } from "vitest";
import { getModel, streamSimple } from "../src/compat.ts";
import type { AssistantMessage, Context, Model, Tool, ToolCall } from "../src/types.ts";

/**
 * Credential-gated live probe for *registration-time* deferral on GitHub Copilot's Anthropic
 * gateway. The sibling probe in `github-copilot-deferred-tools-live.test.ts` covers the
 * transcript-marker path; this one covers turn zero, where there is no transcript at all and the
 * only thing withholding a schema is `Tool.deferred`.
 *
 * Skipped unless GITHUB_COPILOT_TOKEN is set. It makes three small billable requests against
 * claude-opus-5 with a tiny max output, because the reported *input* token count is the signal.
 */

const TOKEN = process.env.GITHUB_COPILOT_TOKEN;
const BASE_URL = process.env.GITHUB_COPILOT_BASE_URL;

/**
 * Each definition carries its own prose rather than a repeated block. Near-identical padding
 * repeated across a large tool list reads as an injection pattern to the gateway's content
 * filter, which then blocks the inline baseline and makes the comparison unmeasurable.
 */
function makeTool(name: string, detail: string, deferred: boolean): Tool {
	return {
		name,
		...(deferred ? { deferred: true } : {}),
		description: detail,
		parameters: {
			type: "object",
			properties: {
				entry: { type: "string", description: `Identifier of the ${name} entry to read.` },
				fields: { type: "array", items: { type: "string" }, description: "Field names to include." },
			},
			required: ["entry"],
			additionalProperties: false,
		},
	};
}

const IMMEDIATE_NAME = "catalogue_index";
const DEFERRED_TOOLS: { name: string; detail: string }[] = [
	{
		name: "catalogue_authors",
		detail:
			"Read one author record from the library catalogue. An author record holds the preferred display name, any alternate spellings recorded by cataloguers over the years, birth and death years where they are known, and a short biographical note written for readers browsing the shelves.",
	},
	{
		name: "catalogue_titles",
		detail:
			"Read one title record. A title record holds the work's main title, any subtitle printed on the title page, the uniform title used to gather translations together, and the language the work was originally written in.",
	},
	{
		name: "catalogue_editions",
		detail:
			"Read one edition record. Editions describe a particular printing: the publisher, the city and year of publication, the page count, the binding, and the identifiers such as ISBN that distinguish it from other printings of the same work.",
	},
	{
		name: "catalogue_subjects",
		detail:
			"Read one subject heading. Subject headings place a work within the classification scheme, listing the broader heading it sits under, the narrower headings beneath it, and the related headings a reader might follow sideways.",
	},
	{
		name: "catalogue_shelves",
		detail:
			"Read one shelf location. A shelf location gives the building, the floor, the range, and the call number span held there, so a reader can walk to the right part of the collection without asking at the desk.",
	},
	{
		name: "catalogue_loans",
		detail:
			"Read one loan record. A loan record shows which copy left the building, the date it was taken out, the date it is due back, and how many times the loan has been renewed by the reader who holds it.",
	},
	{
		name: "catalogue_notes",
		detail:
			"Read one cataloguer's note. Notes carry the observations that do not fit the structured fields: a description of an unusual binding, a record of a previous owner's bookplate, or an explanation of why two similar records were kept apart.",
	},
	{
		name: "catalogue_reviews",
		detail:
			"Read one review summary. Review summaries collect what published reviewers said about a work, when the review appeared, which publication carried it, and a brief extract chosen to convey the reviewer's overall judgement.",
	},
];

const DEFERRED_NAMES = DEFERRED_TOOLS.map((tool) => tool.name);

function makeTools(deferred: boolean): Tool[] {
	return [
		makeTool(
			IMMEDIATE_NAME,
			"List the sections of the library catalogue that are available to read, so a reader knows which section to open next.",
			false,
		),
		...DEFERRED_TOOLS.map((tool) => makeTool(tool.name, tool.detail, deferred)),
	];
}

/** Turn zero: a single user message, no transcript, so only registration can withhold a schema. */
function turnZero(tools: Tool[]): Context {
	return {
		systemPrompt: "You are a terse assistant. Answer in one short sentence and call no tools.",
		tools,
		messages: [{ role: "user", content: "Say ready.", timestamp: Date.now() }],
	};
}

/** A transcript whose tool result activates two registration-deferred tools. */
function afterActivation(tools: Tool[], addedToolNames: string[]): Context {
	return {
		systemPrompt: "You are a terse assistant. Answer in one short sentence and call no tools.",
		tools,
		messages: [
			{ role: "user", content: "Open the catalogue sections.", timestamp: Date.now() },
			{
				role: "assistant",
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
				content: [{ type: "toolCall", id: "call_load", name: IMMEDIATE_NAME, arguments: { entry: "all" } }],
				timestamp: Date.now(),
			},
			{
				role: "toolResult",
				toolCallId: "call_load",
				toolName: IMMEDIATE_NAME,
				content: [{ type: "text", text: `Opened ${addedToolNames.length} sections.` }],
				isError: false,
				addedToolNames,
				timestamp: Date.now(),
			},
			{ role: "user", content: "Say ready.", timestamp: Date.now() },
		],
	};
}

interface AnthropicPayload {
	tools?: { name: string; defer_loading?: boolean }[];
	messages?: { content?: unknown }[];
}

interface ProbeResult {
	promptTokens: number;
	/** Client tool names only; the provider-injected search tool is reported separately. */
	toolNames: string[];
	/** Whether mcpi appended Anthropic's server-side search tool to the request. */
	searchTool: boolean;
	deferred: string[];
	toolReferences: string[];
}

/**
 * Collect every `tool_reference` name the request carries. The blocks are nested inside a
 * `tool_result` block's own content, and they name the tool in `tool_name`.
 */
function toolReferenceNames(payload: AnthropicPayload | undefined): string[] {
	const names: string[] = [];
	const visit = (blocks: unknown): void => {
		if (!Array.isArray(blocks)) return;
		for (const block of blocks) {
			if (!block || typeof block !== "object") continue;
			const candidate = block as { type?: string; tool_name?: string; content?: unknown };
			if (candidate.type === "tool_reference" && typeof candidate.tool_name === "string") {
				names.push(candidate.tool_name);
			}
			visit(candidate.content);
		}
	};
	for (const message of payload?.messages ?? []) visit(message.content);
	return names;
}

async function probe(context: Context): Promise<ProbeResult> {
	// No capability pin: the Copilot catalog now advertises deferral for this model, so the probe
	// exercises the shipped default rather than a forced one.
	const base = getModel("github-copilot", "claude-opus-5");
	const model: Model<"anthropic-messages"> = { ...base, ...(BASE_URL ? { baseUrl: BASE_URL } : {}) };
	let payload: AnthropicPayload | undefined;
	const s = streamSimple(model, context, {
		apiKey: TOKEN,
		maxTokens: 16,
		onPayload: (p) => {
			payload = p as AnthropicPayload;
			return p;
		},
	});
	for await (const _ of s) {
		// Drain the stream; only the final usage matters.
	}
	const response = await s.result();
	expect(response.errorMessage).toBeFalsy();
	// The gateway caches the prompt, so the tool definitions land in cacheWrite on the first
	// request and cacheRead afterwards. Only the sum tracks what the request actually carried.
	const { input, cacheRead, cacheWrite } = response.usage;
	return {
		promptTokens: input + cacheRead + cacheWrite,
		toolNames: (payload?.tools ?? []).map((t) => t.name).filter((name) => !name.startsWith("tool_search_tool")),
		searchTool: (payload?.tools ?? []).some((t) => t.name.startsWith("tool_search_tool")),
		deferred: (payload?.tools ?? []).filter((t) => t.defer_loading).map((t) => t.name),
		toolReferences: toolReferenceNames(payload),
	};
}

/** Like `probe`, but keeps the assistant reply so a second turn can build on it. */
async function call(
	context: Context,
): Promise<{ message: AssistantMessage; payload: AnthropicPayload | undefined; calls: ToolCall[] }> {
	const base = getModel("github-copilot", "claude-opus-5");
	const model: Model<"anthropic-messages"> = { ...base, ...(BASE_URL ? { baseUrl: BASE_URL } : {}) };
	let payload: AnthropicPayload | undefined;
	const s = streamSimple(model, context, {
		apiKey: TOKEN,
		maxTokens: 1024,
		onPayload: (p) => {
			payload = p as AnthropicPayload;
			return p;
		},
	});
	for await (const _ of s) {
		// Drain the stream.
	}
	const message = await s.result();
	expect(message.errorMessage).toBeFalsy();
	return { message, payload, calls: message.content.filter((entry) => entry.type === "toolCall") };
}

describe.skipIf(!TOKEN)("GitHub Copilot registration deferral (live)", () => {
	it(
		"defers registered schemas on turn zero and loads them at an activation marker",
		{ retry: 2, timeout: 120000 },
		async () => {
			// Turn zero, no transcript: registration alone withholds the schemas, and the tools
			// array keeps every name in registration order so dispatch and grammar are unchanged.
			const deferredRun = await probe(turnZero(makeTools(true)));
			expect(deferredRun.toolNames).toEqual([IMMEDIATE_NAME, ...DEFERRED_NAMES]);
			expect(deferredRun.deferred).toEqual(DEFERRED_NAMES);
			expect(deferredRun.toolReferences).toEqual([]);
			// Sent alongside the deferred definitions, so a deferred tool no marker names is still
			// reachable. Nothing else in this file would notice if it stopped being appended.
			expect(deferredRun.searchTool).toBe(true);

			// An `addedToolNames` marker activates two of them at that tool-result position. The
			// rest stay deferred and the tools array is unchanged, so the cacheable prefix holds.
			const activated = DEFERRED_NAMES.slice(0, 2);
			const activation = await probe(afterActivation(makeTools(true), activated));
			expect(activation.toolNames).toEqual([IMMEDIATE_NAME, ...DEFERRED_NAMES]);
			expect(activation.deferred).toEqual(DEFERRED_NAMES);
			expect(activation.toolReferences).toEqual(activated);

			// Same tools without the registration flag: every schema is inlined on turn zero.
			const inline = await probe(turnZero(makeTools(false)));
			expect(inline.deferred).toEqual([]);
			expect(inline.toolNames).toEqual([IMMEDIATE_NAME, ...DEFERRED_NAMES]);
			// Nothing is held back, so there is no catalog to search and no search tool to send.
			expect(inline.searchTool).toBe(false);

			// A conservative floor, so the test tracks the capability rather than a tokenizer
			// revision. The measured turn-zero pair was 1,650 inline against 543 deferred.
			expect(inline.promptTokens - deferredRun.promptTokens).toBeGreaterThan(500);
		},
	);

	/**
	 * Reachability. A registration-deferred tool that no skill ever mentions must still be usable.
	 * Anthropic's `defer_loading` hides a schema from the model but provides no discovery of its
	 * own unless the request also carries a tool-search server tool, which mcpi does not send. The
	 * always-immediate search tool is what makes deferred tools reachable: the model calls it, the
	 * result names a tool through `addedToolNames`, and the schema arrives as a `tool_reference`.
	 *
	 * This depends on the model choosing to search, so it is a live behavioral check rather than a
	 * wire-shape assertion; the deterministic shape is covered offline in
	 * `registration-deferred-tools.test.ts`.
	 */
	it("reaches a deferred tool no skill named, through the search tool", { retry: 2, timeout: 180000 }, async () => {
		const tools = makeTools(true);
		const target = "catalogue_reviews";
		const system =
			`You are a librarian. Only ${IMMEDIATE_NAME} is loaded right now. Call it first to find out which ` +
			"section handles the reader's question, then call the section tool it names. Always use a tool.";
		const question = "What have reviewers written about 'Dune'?";
		const discovery: Context = {
			systemPrompt: system,
			messages: [{ role: "user", content: question, timestamp: 1 }],
			tools,
		};
		const found = await call(discovery);
		expect(found.calls.map((entry) => entry.name)).toContain(IMMEDIATE_NAME);

		const searchCall = found.calls.find((entry) => entry.name === IMMEDIATE_NAME);
		const used = await call({
			systemPrompt: system,
			messages: [
				{ role: "user", content: question, timestamp: 1 },
				found.message,
				{
					role: "toolResult",
					toolCallId: searchCall?.id ?? "call_1",
					toolName: IMMEDIATE_NAME,
					content: [{ type: "text", text: `Reader reviews are handled by ${target}. Call it now.` }],
					addedToolNames: [target],
					isError: false,
					timestamp: 3,
				},
			],
			tools,
		});

		// Loaded by a generic search result, not by a skill, and callable immediately after.
		expect(toolReferenceNames(used.payload)).toEqual([target]);
		expect(used.calls.map((entry) => entry.name)).toContain(target);
	});
});
