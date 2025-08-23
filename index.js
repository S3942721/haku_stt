import { BedrockAgentRuntimeClient, RetrieveCommand } from "@aws-sdk/client-bedrock-agent-runtime"
import { BedrockRuntimeClient, ConverseStreamCommand } from "@aws-sdk/client-bedrock-runtime"
import { ApiGatewayManagementApiClient, PostToConnectionCommand } from "@aws-sdk/client-apigatewaymanagementapi"

const bedrockAgentClient = new BedrockAgentRuntimeClient({})
const bedrockClient = new BedrockRuntimeClient({ region: process.env.MODEL_REGION })

console.log('AWS Region:', process.env.AWS_REGION)
console.log('Bedrock client region:', bedrockClient.config.region)
const apigwManagementApi = new ApiGatewayManagementApiClient({
  endpoint: process.env.API_GATEWAY_ENDPOINT
})

// Validate environment variables
if (!process.env.API_GATEWAY_ENDPOINT) {
  console.warn('WARNING: API_GATEWAY_ENDPOINT environment variable is not set')
}

if (!process.env.BEDROCK_MODEL_ID) {
  console.warn('WARNING: BEDROCK_MODEL_ID environment variable is not set')
}

const RMIT_INFO_ONLY = + (process.env.RMIT_INFO_ONLY ?? 1)

async function handleKnowledgeBaseSearch ({ query }) {
  if (!query) {
    return {
      error: 'Query parameter is required'
    }
  }

  const knowledgeBaseId = process.env.KB_ID

  const input = {
    knowledgeBaseId,
    retrievalQuery: {
      text: query
    },
    retrievalConfiguration: {
      vectorSearchConfiguration: {
        numberOfResults: 5
      }
    }
  }

  try {
    const command = new RetrieveCommand(input)
    const response = await bedrockAgentClient.send(command)

    const rawResults = await Promise.all(response.retrievalResults.map(async (result) => {
      const { score, content, metadata } = result
      const { text } = content

      let sourceUrl = null
      try {
        sourceUrl = metadata?.['source-url'] || null

        if (!sourceUrl && metadata?.['x-amz-bedrock-kb-source-uri']) {
          const s3Uri = metadata['x-amz-bedrock-kb-source-uri']
          console.log('Found S3 URI:', s3Uri)

          const s3Match = s3Uri.match(/s3:\/\/([^\/]+)\/(.+)/)
          if (s3Match) {
            const [, bucket, key] = s3Match
            try {
              const { HeadObjectCommand, S3Client } = await import('@aws-sdk/client-s3')
              const s3Client = new S3Client({ region: process.env.AWS_REGION })

              const headResponse = await s3Client.send(new HeadObjectCommand({
                Bucket: bucket,
                Key: decodeURIComponent(key)
              }))

              sourceUrl = headResponse.Metadata?.['source-url'] || null
              console.log('Retrieved source-url from S3:', sourceUrl)
            } catch (s3Error) {
              console.error('Error fetching S3 metadata:', s3Error)
            }
          }
        }
      } catch (e) {
        console.warn('Error extracting metadata:', e)
      }

      return {
        score: +score.toFixed(2),
        text,
        sourceUrl,
        s3Uri: metadata?.['x-amz-bedrock-kb-source-uri'] || null,
        summary: score < 0.3 ? "⚠️ Very low relevance" : score < 0.5 ? "🔍 Somewhat related" : undefined
      }
    }))

    const filteredResults = rawResults.filter(r => r.score >= 0.3)

    if (filteredResults.length === 0) {
      return [{
        score: 0.0,
        text: "No relevant information found in the Knowledge Base.",
        summary: "No match found — consider using WebSearch instead."
      }]
    }

    return filteredResults
  } catch (error) {
    console.error('KnowledgeBase search error:', error)
    return [{
      score: 0.0,
      text: "An error occurred while searching the Knowledge Base.",
      summary: error.message
    }]
  }
}

async function handleToolCall ({ name, input }) {
  try {
    switch (name) {
      case "KnowledgeBaseSearch":
        return await handleKnowledgeBaseSearch(input)
      default:
        return {
          error: `Unknown Tool Name: ${name}`
        }
    }
  } catch (error) {
    console.error('Error in handler:', error)
    return {
      error: error.message
    }
  }
}

const SYSTEM_PROMPT_WITHOUT_WEBSEARCH =
  `# Instructions

You are a helpful assistant at RMIT University Researcher Portal designed to help users with their research questions. You have access to internal RMIT University Researcher Portal data in your knowledge base.

When responding to student queries:

1. Synthesize the information into a clear, helpful response, DONOT mention what's already in your record if you have gathered updated information
2. Always cite your sources when providing information
3. If you don't know the answer or can't find relevant information, be honest about your limitations
4. If you tried search for three times and haven't found any relevant results, reply that you couldn't find information.

# Basic Information

Currently it is 2025; You are in Australia.
If your search result contains a link with only article id (e.g. article?id=xxx), the URL would be \`https://rmitheda.my.site.com/Researcherportal/s/article?id=xxx\`, so make sure you provide the correct article link to users as they will have access to the portal.

# Response Format

Your responses should be:
- ALWAYS wrapped by <Thinking></Thinking> or <Response></Response> tags, remember everything not included in those blocks will be deprecated.
  - If you think there's a tool call required, wrap your response in <Thinking></Thinking> block even you can see the tag in history, or after a tool calling etc.
  - If you think you've got enough information to answer, wrap your response in <Response></Response> block.
  - Content wrapped by <Thinking></Thinking> or <Response></Response> should be in markdown format. Any link should be wrapped by []();
- Clear and concise
- Well-structured with appropriate headings and bullet points when needed
- Written in a helpful, supportive tone
- Free of jargon unless necessary for the subject matter
- Properly cited with sources if they came from internet.

## Example Response

User: What is ARC DECRA?
Assistant: <Thinking>The user asks something about ARC DECRA, that information might inside knowledge base, let me search it.</Thinking>
User: <tool-results-with-score-and-text>
Assistant: <Thinking>Now I've got enough information to answer this, let me parse my response</Thinking><Response>Here's information about ARC DECRA</Response>

Remember that your goal is to provide accurate, helpful information to support students in their academic journey.
`

const SYSTEM_PROMPT_QUICK = `# Instructions

You are a fast and concise assistant for the RMIT Researcher Portal.

In **Quick Mode**, your goal is to answer as efficiently as possible.  
You must follow these rules:

- Keep your responses short and direct — typically no more than 1-3 sentences.
- Only respond based on what you already know or from the knowledge base.
- Do **not** use WebSearch or any online search tools.
- You may use the internal RMIT Knowledge Base to assist your answer.
- If the first search returns no relevant results, you may attempt one more refined query.
- Do **not** summarize multiple perspectives or cite external sources.

# When generating responses:
- Use <Response>...</Response> tags and markdown formatting.
- Skip background information or disclaimers.
- Cite only one relevant source if available (e.g., _Source: [Title](URL)_).
- Be direct and helpful. If unsure, reply: "I don't have enough information to answer that."

# Note

Quick Mode responses prioritize **speed and clarity**, but you may call KnowledgeBaseSearch tools up to 2 times if needed.
You must never use WebSearch or return information retrieved from the internet.
`


const ENABLE_REALTIME_SEARCH = + process.env.ENABLE_REALTIME_SEARCH || 0
const SYSTEM_PROMPT = ENABLE_REALTIME_SEARCH ? SYSTEM_PROMPT_WITH_WEBSEARCH : SYSTEM_PROMPT_WITHOUT_WEBSEARCH

const KNOWLEDGE_BASE_TOOL = {
  toolSpec: {
    name: "KnowledgeBaseSearch",
    description: "Search knowledge base using search engine",
    inputSchema: {
      json: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query to be used."
          }
        },
        required: ["query"]
      }
    }
  }
}

const TOOLS = [KNOWLEDGE_BASE_TOOL]

let connectionId = '', sessionId = ''

// send chunks through API Gateway Websocket connection
async function sendChunk (data) {
  try {
    // Check if API Gateway endpoint is configured
    if (!process.env.API_GATEWAY_ENDPOINT) {
      console.error('API_GATEWAY_ENDPOINT environment variable is not set')
      throw new Error('API Gateway endpoint not configured')
    }

    const command = new PostToConnectionCommand({
      ConnectionId: connectionId,
      Data: JSON.stringify({ action: "completion", sessionId, ...data })
    })
    await apigwManagementApi.send(command)
  } catch (error) {
    if (error.$metadata?.httpStatusCode === 410) {
      console.log('Connection is no longer available')
    } else {
      console.error('Failed to send message:', error)
      // Don't throw here to allow the function to continue even if sending fails
    }
  }
}

function getInput (messages, disableTools = false, deepSearch = false) {
  const input = {
    modelId: process.env.BEDROCK_MODEL_ID,
    inferenceConfig: {
      temperature: 0.2,
      topP: 0.9
    },
    messages,
    system: [
      {
        text: deepSearch
          ? SYSTEM_PROMPT_WITHOUT_WEBSEARCH
          : SYSTEM_PROMPT_QUICK
      }
    ]
  }

  if (!disableTools) {
    input.toolConfig = {
      tools: deepSearch
        ? TOOLS // In Deep Search mode, both KnowledgeBaseSearch and WebSearch are available
        : [KNOWLEDGE_BASE_TOOL] // In Quick Mode, only KnowledgeBaseSearch is available
    }
  }

  return input
}

async function invokeBedrockModel (messages, prompt, chunkCount, disableTools = false, quickMode = false) {
  console.log("input.messages before model call:\\n", JSON.stringify(messages, null, 2))
  console.log("invokeBedrockModel - BEDROCK_MODEL_ID:", process.env.BEDROCK_MODEL_ID)
  try {
    const input = getInput(messages, disableTools, quickMode)
    console.log("invokeBedrockModel - input.modelId:", input.modelId)
    if (prompt) {
      const userMessageWithToolResult = input.messages.pop()
      userMessageWithToolResult.content.push({ text: prompt })
      input.messages.push(userMessageWithToolResult)
    }

    let toolUse = null, responseText = ''

    const command = new ConverseStreamCommand(input)
    const response = await bedrockClient.send(command)

    for await (const chunk of response.stream) {
      if (chunk.contentBlockStart?.start?.toolUse) {
        toolUse = { ...chunk.contentBlockStart.start.toolUse, input: '' }
      }

      if (chunk.contentBlockDelta) {
        const delta = chunk.contentBlockDelta.delta
        if (delta.text) {
          responseText += delta.text
          sendChunk({
            content: delta.text,
            isFinished: false,
            chunkNumber: chunkCount++
          })
        } else if (delta.toolUse) {
          const inputChunk = delta.toolUse.input
          toolUse && (toolUse.input += inputChunk)
        }
      } else if (chunk.messageStop) {
        if (toolUse) {
          console.log(`Handling tool call: ${toolUse.name} input: ${toolUse.input}`)

          await sendChunk({
            content: `<Thinking><ToolUse data-name="${toolUse.name}" data-id="${toolUse.toolUseId}" data-content="${toolUse.input?.replace(/'/g, "<REPLACE_APOSTROPHE>").replace(/"/g, "'")}" /></Thinking>`,
            isFinished: false,
            chunkNumber: chunkCount++
          })

          try {
            if (disableTools) {
              console.warn('⚠️ Tool use received but tools are disabled, skipping tool execution.')
              return { isFinished: false, chunkCount }
            }

            toolUse.input = JSON.parse(toolUse.input)

            // responseText is sent first as Thinking
            if (responseText) {
              messages.push({
                role: 'assistant',
                content: [{ text: responseText }]
              })
            }

            // toolUse is a separate message (must be in the last assistant message)
            messages.push({
              role: 'assistant',
              content: [{ toolUse }]
            })

            const result = await handleToolCall(toolUse)
            console.log('Tool result:', JSON.stringify(result, null, 2))

            try {
              if (toolUse.name === 'KnowledgeBaseSearch' && Array.isArray(result)) {

                for (let i = 0; i < result.length; i++) {
                  const r = result[i]
                  console.log(`Result ${i}: score=${r.score}, hasSourceUrl=${!!r.sourceUrl}, sourceUrl=${r.sourceUrl}`)

                  if (r.sourceUrl && r.score >= 0.5) {
                    let title = 'Document'
                    if (r.s3Uri) {
                      const fileName = r.s3Uri.split('/').pop()
                      title = fileName.replace(/\.[^/.]+$/, '')
                    }

                    const originalLength = r.text.length
                    result[i].text = r.text + `\n\n_Source: [${title}](${r.sourceUrl})_`
                  } else {
                    console.log(`Skipped result ${i}: score too low or no sourceUrl`)
                  }
                }
              }
            } catch (sourcesError) {
              console.warn('Error processing sources:', sourcesError)
            }

            messages.push({
              role: 'user',
              content: [{
                toolResult: {
                  toolUseId: toolUse.toolUseId,
                  content: [{ json: { results: result } }]
                }
              }]
            })

            if (toolUse.name === 'WebSearch') {
              const contentObject = {
                query: toolUse.input.query,
                results: result.results
              }

              const contentString = JSON.stringify(contentObject)

              await sendChunk({
                content: `<Thinking><ToolUse data-name="WebsearchResult" data-id="${toolUse.toolUseId}" data-content='${contentString.replace(/'/g, "&apos;")}' /></Thinking>`,
                isFinished: false,
                chunkNumber: chunkCount++
              })
            }

            return { isFinished: false, chunkCount }
          } catch (toolError) {
            console.error('Error processing tool call:', toolError)
            messages.push({
              role: 'user',
              content: [{
                toolResult: {
                  toolUseId: toolUse.toolUseId,
                  content: [{ json: { results: { error: 'Tool processing error' } } }]
                }
              }]
            })
            return { isFinished: false, chunkCount }
          }
        } else {
          sendChunk({ isFinished: true, chunkNumber: chunkCount++ })
          return { isFinished: true, chunkCount }
        }
      }
    }

    sendChunk({ isFinished: true, chunkNumber: chunkCount++ })
    return { isFinished: true, chunkCount }
  } catch (error) {
    console.error('Error:', error)

    if (error.name === 'ServiceUnavailableException' || error?.$metadata?.httpStatusCode === 503) {
      await sendChunk({
        content: `<Response>The Bedrock model is currently unable to handle the request. The system may be under high load or under maintenance. Please try again later.</Response>`,
        isFinished: true,
        chunkNumber: chunkCount++
      })
      return { isFinished: true, chunkCount }
    }

    // Default error handling
    await sendChunk({
      content: `<Response> An unknown error occurred:${error.message}</Response>`,
      isFinished: true,
      chunkNumber: chunkCount++
    })
    return { isFinished: true, chunkCount }
  }
}

async function startCompletion ({ history, deepSearch = false }) {
  try {
    sessionId = Math.random().toString(32).slice(2)
    const maxIterations = process.env.MAX_ITERATIONS || 5
    // Initialize messages array from history if needed
    let messages = history.map(({ role, content }) => {
      return {
        role,
        content: Array.isArray(content) ? content : [{ text: content }]
      }
    })

    let iteration = 0, chunkCount = 0
    while (++iteration <= maxIterations) {
      const disableTools = iteration === maxIterations

      const { isFinished, chunkCount: newChunkCount } = await invokeBedrockModel(
        messages,
        disableTools && "Please provide final answer...",
        chunkCount,
        disableTools,
        deepSearch
      )
      chunkCount = newChunkCount
      if (isFinished) {
        break
      }
    }

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: 'Successfully processed request'
      })
    }

  } catch (error) {
    console.error('Error:', error)
    sendChunk({ error: error.message, isFinished: true })
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: 'Internal server error',
        error: error.message
      })
    }
  }
}

export const handler = async (event) => {
  const { requestContext, body, headers } = event
  const { routeKey } = requestContext

  if (routeKey === '$connect') {
    const origin = headers.Origin || headers.origin
    const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || null
    if (allowedOrigins && !allowedOrigins.includes(origin)) {
      return {
        statusCode: 403,
        body: 'Forbidden'
      }
    }
    return {
      statusCode: 200,
      body: 'Connected'
    }
  }
  connectionId = requestContext.connectionId

  const jsonBody = JSON.parse(body)
  const { action, deepSearch = false } = jsonBody

  switch (action) {
    case 'completion':
      return await startCompletion({ ...jsonBody, deepSearch })
  }
}