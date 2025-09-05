import { randomUUID } from "@copilotkit/shared";
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

  constructor(config: LangGraphServiceAdapterConfig) {
    this.agent = config.agent;
    this.debug = config.debug || false;
  }

  async process(
    request: CopilotRuntimeChatCompletionRequest,
  ): Promise<CopilotRuntimeChatCompletionResponse> {
    const { eventSource, messages, actions, threadId, runId } = request;

    try {
      // Convert CopilotKit format to LangGraph input
      const langGraphInput = convertCopilotKitToLangGraphInput({
        messages,
        actions,
        threadId,
        runId,
      });

      // Process event stream
      eventSource.stream(async (eventStream$) => {
        await this.processLangGraphStream(langGraphInput, eventStream$);
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
    eventStream$: RuntimeEventSubject,
  ): Promise<void> {
    // Create stream state for managing message and tool call states
    const streamState = createStreamState();

    try {
      // Get event stream from LangGraph agent
      const eventStream = this.agent.streamEvents(input, {
        version: "v2",
      });

      // Process each event using ported CopilotKit LangGraphAgent logic
      for await (const event of eventStream) {
        await handleLangGraphEvent(
          event,
          eventStream$,
          streamState,
          this.debug,
        );
      }
    } catch (error) {
      if (this.debug) {
        console.error("[LangGraph] Error during stream processing:", error);
      }
      throw convertServiceAdapterError(error, "LangGraph");
    } finally {
      eventStream$.complete();
    }
  }
}
