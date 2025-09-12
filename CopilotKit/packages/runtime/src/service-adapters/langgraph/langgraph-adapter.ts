import { randomUUID } from "@copilotkit/shared";
import { isSystemMessage } from "@langchain/core/messages";
import type {
  BaseChannel,
  PregelNode,
  PregelOptions,
} from "@langchain/langgraph";
import type { RuntimeEventSubject } from "../events";
import type {
  CopilotRuntimeChatCompletionRequest,
  CopilotRuntimeChatCompletionResponse,
  CopilotServiceAdapter,
} from "../service-adapter";
import { convertServiceAdapterError } from "../shared";
import type { LangGraphInput, LangGraphServiceAdapterConfig } from "./types";
import {
  convertCopilotKitToLangGraphInput,
  createStreamState,
  handleLangGraphEvent,
} from "./utils";

/**
 * LangGraph ServiceAdapter for CopilotKit Runtime
 *
 * This adapter integrates local LangGraph agents (CompiledStateGraph) with CopilotKit,
 * providing full observability of agent execution through event streaming.
 *
 * Unlike the existing LangGraphAgent which works with remote LangGraph Platform,
 * this adapter works with local LangGraph agent instances.
 */
export class LangGraphServiceAdapter implements CopilotServiceAdapter {
  private agent: LangGraphServiceAdapterConfig["agent"];
  private debug: boolean;
  private systemPromptStrategy: "passthrough" | "inject";

  constructor(config: LangGraphServiceAdapterConfig) {
    this.agent = config.agent;
    this.debug = config.debug || false;
    this.systemPromptStrategy = config.systemPromptStrategy ?? "passthrough";

    if (this.debug) {
      console.log("[DEBUG] LangGraphServiceAdapter created with config:", {
        debug: this.debug,
        systemPromptStrategy: this.systemPromptStrategy,
      });
      console.log("[DEBUG] Agent type:", this.agent.constructor.name);
    }
  }

  async process(
    request: CopilotRuntimeChatCompletionRequest,
  ): Promise<CopilotRuntimeChatCompletionResponse> {
    const { eventSource, messages, actions, threadId, runId } = request;

    try {
      // First, convert the entire input to LangChain format.
      const langGraphInput = convertCopilotKitToLangGraphInput({
        messages,
        actions,
        debug: this.debug,
      });

      const streamConfig: Partial<
        PregelOptions<Record<string, PregelNode>, Record<string, BaseChannel>>
      > & { version: "v1" | "v2" } = { version: "v2" };

      // Decide which strategy to use for handling the system prompt.
      if (this.systemPromptStrategy === "inject") {
        let copilotkitInstructions = "";
        // Filter out the system message and store its content.
        const filteredMessages = langGraphInput.messages.filter((msg) => {
          if (isSystemMessage(msg)) {
            copilotkitInstructions = msg.content as string;
            return false; // Remove from message list
          }
          return true;
        });

        // Update the input with the filtered messages.
        langGraphInput.messages = filteredMessages;
        // Add the extracted instructions to the config for injection.
        streamConfig.configurable = {
          copilotkit_instructions: copilotkitInstructions,
        };

        if (this.debug) {
          console.log(
            "[DEBUG] 'inject' strategy active. Instructions extracted and messages filtered.",
          );
        }
      } else {
        // For 'passthrough' strategy, do nothing. The input is used as is.
        if (this.debug) {
          console.log(
            "[DEBUG] 'passthrough' strategy active. Passing all messages directly.",
          );
        }
      }

      // Process event stream
      eventSource.stream(async (eventStream$) => {
        await this.processLangGraphStream(
          langGraphInput,
          streamConfig,
          eventStream$,
        );
      });

      return {
        threadId: threadId || randomUUID(),
        runId,
      };
    } catch (error) {
      throw convertServiceAdapterError(error, "LangGraph");
    }
  }

  private async processLangGraphStream(
    input: LangGraphInput,
    config: Partial<
      PregelOptions<Record<string, PregelNode>, Record<string, BaseChannel>>
    > & {
      version: "v1" | "v2";
    },
    eventStream$: RuntimeEventSubject,
  ): Promise<void> {
    if (this.debug) {
      console.log("[DEBUG] === processLangGraphStream START ===");
      console.log("[DEBUG] LangGraph input:", {
        messagesCount: input.messages.length,
        toolsCount: input.tools.length,
        messages: input.messages.map((msg) => ({
          type: msg.constructor.name,
          content:
            typeof msg.content === "string"
              ? `${msg.content.substring(0, 100)}...`
              : msg.content,
        })),
        tools: input.tools.map((tool) => tool.name),
      });
      if (config.configurable) {
        console.log(
          "[DEBUG] Injecting configurable fields:",
          Object.keys(config.configurable),
        );
      }
    }

    const streamState = createStreamState();

    try {
      const eventStream = this.agent.streamEvents(input, config);

      if (this.debug) {
        console.log(
          "[DEBUG] StreamEvents created successfully, starting iteration...",
        );
      }

      let eventCount = 0;
      const eventTypes = new Map<string, number>();

      for await (const event of eventStream) {
        eventCount++;
        const eventType = event.event;
        eventTypes.set(eventType, (eventTypes.get(eventType) || 0) + 1);

        if (this.debug) {
          console.log(`[DEBUG] Event ${eventCount} (${eventType}):`, {
            event: event.event,
            run_id: event.run_id,
            metadata: event.metadata,
            data: event.data?.chunk
              ? {
                  chunk: {
                    id: event.data.chunk.id,
                    content: event.data.chunk.content,
                    tool_calls: event.data.chunk.tool_calls,
                    tool_call_chunks: event.data.chunk.tool_call_chunks,
                  },
                }
              : event.data,
          });
        }

        await handleLangGraphEvent(
          event,
          eventStream$,
          streamState,
          this.debug,
        );
      }

      if (this.debug) {
        console.log(`[DEBUG] Total events processed: ${eventCount}`);
        console.log(
          "[DEBUG] Event type distribution:",
          Object.fromEntries(eventTypes),
        );

        if (eventCount === 0) {
          console.warn(
            "[DEBUG] ⚠️  NO EVENTS PRODUCED! This might indicate an input format problem.",
          );
        }

        const chatModelStreamCount =
          eventTypes.get("on_chat_model_stream") || 0;
        if (chatModelStreamCount === 0) {
          console.warn(
            "[DEBUG] ⚠️  NO on_chat_model_stream EVENTS! This explains why there are no TextMessage events.",
          );
        } else {
          console.log(
            `[DEBUG] ✅ Found ${chatModelStreamCount} on_chat_model_stream events`,
          );
        }

        console.log("[DEBUG] === processLangGraphStream END ===");
      }
    } catch (error) {
      if (this.debug) {
        console.error("[DEBUG] Error in processLangGraphStream:", error);
        console.error("[LangGraph] Error during stream processing:", error);
      }
      throw convertServiceAdapterError(error, "LangGraph");
    } finally {
      eventStream$.complete();
    }
  }
}
