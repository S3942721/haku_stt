"""
Lambda LLM Client - Python implementation of the JavaScript lambda function
Provides direct integration with AWS Bedrock and Knowledge Base functionality
"""

import json
import os
import boto3
import asyncio
import re
from typing import Dict, List, Any, Optional
from botocore.exceptions import ClientError

class LambdaLLMClient:
    """Direct implementation of the lambda function's LLM functionality"""
    
    def __init__(self, region: str = "us-east-1", model_id: str = None):
        """Initialize the Lambda LLM client with AWS services"""
        self.region = region
        self.model_id = model_id or os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        self.kb_id = os.getenv('KB_ID')
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', '5'))
        
        # Initialize AWS clients
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region)
        self.bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=region)
        
        # System prompts
        self.system_prompt_quick = """# Instructions

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
"""
        
        # Tool configuration
        self.knowledge_base_tool = {
            "toolSpec": {
                "name": "KnowledgeBaseSearch",
                "description": "Search knowledge base using search engine",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to be used."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        }
        
        self.conversation_history = []
        self.session_id = None
        
        print(f"Lambda LLM Client initialized - Region: {region}, Model: {self.model_id}")
    
    async def handle_knowledge_base_search(self, query: str) -> List[Dict]:
        """Handle knowledge base search (equivalent to JavaScript handleKnowledgeBaseSearch)"""
        if not query:
            return [{"error": "Query parameter is required"}]
        
        if not self.kb_id:
            return [{"error": "Knowledge Base ID not configured"}]
        
        try:
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5
                    }
                }
            )
            
            raw_results = []
            for result in response['retrievalResults']:
                score = float(result['score'])
                text = result['content']['text']
                metadata = result.get('metadata', {})
                
                source_url = metadata.get('source-url')
                s3_uri = metadata.get('x-amz-bedrock-kb-source-uri')
                
                # Try to get source URL from S3 metadata if not present
                if not source_url and s3_uri:
                    try:
                        # Parse S3 URI and get metadata
                        s3_match = re.match(r's3://([^/]+)/(.+)', s3_uri)
                        if s3_match:
                            bucket, key = s3_match.groups()
                            s3_client = boto3.client('s3', region_name=self.region)
                            
                            head_response = s3_client.head_object(
                                Bucket=bucket,
                                Key=key.replace('%20', ' ')  # URL decode
                            )
                            source_url = head_response.get('Metadata', {}).get('source-url')
                    except Exception as s3_error:
                        print(f"Error fetching S3 metadata: {s3_error}")
                
                raw_results.append({
                    "score": round(score, 2),
                    "text": text,
                    "sourceUrl": source_url,
                    "s3Uri": s3_uri,
                    "summary": "⚠️ Very low relevance" if score < 0.3 else "🔍 Somewhat related" if score < 0.5 else None
                })
            
            # Filter results by score
            filtered_results = [r for r in raw_results if r['score'] >= 0.3]
            
            if not filtered_results:
                return [{
                    "score": 0.0,
                    "text": "No relevant information found in the Knowledge Base.",
                    "summary": "No match found — consider using WebSearch instead."
                }]
            
            return filtered_results
            
        except Exception as error:
            print(f"Knowledge Base search error: {error}")
            return [{
                "score": 0.0,
                "text": "An error occurred while searching the Knowledge Base.",
                "summary": str(error)
            }]
    
    async def handle_tool_call(self, tool_name: str, tool_input: Dict) -> Any:
        """Handle tool calls (equivalent to JavaScript handleToolCall)"""
        try:
            if tool_name == "KnowledgeBaseSearch":
                return await self.handle_knowledge_base_search(tool_input.get('query', ''))
            else:
                return {"error": f"Unknown Tool Name: {tool_name}"}
        except Exception as error:
            print(f"Error in tool handler: {error}")
            return {"error": str(error)}
    
    def get_input_config(self, messages: List[Dict], disable_tools: bool = False) -> Dict:
        """Get input configuration for Bedrock (equivalent to JavaScript getInput)"""
        input_config = {
            "modelId": self.model_id,
            "inferenceConfig": {
                "temperature": 0.2,
                "topP": 0.9
            },
            "messages": messages,
            "system": [{"text": self.system_prompt_quick}]
        }
        
        if not disable_tools:
            input_config["toolConfig"] = {
                "tools": [self.knowledge_base_tool]
            }
        
        return input_config
    
    async def invoke_bedrock_model(self, messages: List[Dict], prompt: str = None, disable_tools: bool = False) -> str:
        """Invoke Bedrock model with streaming (simplified version of JavaScript invokeBedrockModel)"""
        print(f"Invoking Bedrock model: {self.model_id}")
        
        try:
            input_config = self.get_input_config(messages, disable_tools)
            
            if prompt:
                # Add prompt to the last user message
                if messages and messages[-1]['role'] == 'user':
                    messages[-1]['content'].append({"text": prompt})
                else:
                    messages.append({
                        "role": "user",
                        "content": [{"text": prompt}]
                    })
                input_config["messages"] = messages
            
            # Use converse_stream for streaming response
            response = self.bedrock_client.converse_stream(**input_config)
            
            complete_response = ""
            tool_use = None
            
            for chunk in response['stream']:
                if 'contentBlockStart' in chunk:
                    if 'toolUse' in chunk['contentBlockStart']['start']:
                        tool_use = chunk['contentBlockStart']['start']['toolUse'].copy()
                        tool_use['input'] = ''
                
                elif 'contentBlockDelta' in chunk:
                    delta = chunk['contentBlockDelta']['delta']
                    
                    if 'text' in delta:
                        complete_response += delta['text']
                    elif 'toolUse' in delta and tool_use:
                        tool_use['input'] += delta['toolUse']['input']
                
                elif 'messageStop' in chunk:
                    if tool_use:
                        print(f"Handling tool call: {tool_use['name']}")
                        
                        if disable_tools:
                            print("Tools disabled, skipping tool execution")
                            break
                        
                        # Parse tool input
                        try:
                            tool_input = json.loads(tool_use['input'])
                        except json.JSONDecodeError:
                            tool_input = {"query": tool_use['input']}
                        
                        # Add assistant message with tool use
                        if complete_response:
                            messages.append({
                                "role": "assistant",
                                "content": [{"text": complete_response}]
                            })
                        
                        messages.append({
                            "role": "assistant",
                            "content": [{"toolUse": tool_use}]
                        })
                        
                        # Execute tool
                        tool_result = await self.handle_tool_call(tool_use['name'], tool_input)
                        
                        # Process sources for knowledge base results
                        if tool_use['name'] == 'KnowledgeBaseSearch' and isinstance(tool_result, list):
                            for i, result in enumerate(tool_result):
                                if result.get('sourceUrl') and result.get('score', 0) >= 0.5:
                                    title = 'Document'
                                    if result.get('s3Uri'):
                                        filename = result['s3Uri'].split('/')[-1]
                                        title = filename.replace(r'\.[^.]+$', '')
                                    
                                    tool_result[i]['text'] += f"\n\n_Source: [{title}]({result['sourceUrl']})_"
                        
                        # Add tool result as user message
                        messages.append({
                            "role": "user",
                            "content": [{
                                "toolResult": {
                                    "toolUseId": tool_use['toolUseId'],
                                    "content": [{"json": {"results": tool_result}}]
                                }
                            }]
                        })
                        
                        # Continue with another iteration
                        return await self.invoke_bedrock_model(messages, None, disable_tools)
                    
                    break
            
            return complete_response
            
        except Exception as error:
            print(f"Bedrock model error: {error}")
            
            if hasattr(error, 'response') and error.response.get('Error', {}).get('Code') == 'ServiceUnavailableException':
                return "<Response>The Bedrock model is currently unable to handle the request. The system may be under high load or under maintenance. Please try again later.</Response>"
            
            return f"<Response>An error occurred: {str(error)}</Response>"
    
    async def start_completion(self, history: List[Dict]) -> str:
        """Start completion process (equivalent to JavaScript startCompletion)"""
        try:
            import random
            self.session_id = str(random.randint(100000, 999999))
            
            # Convert history to Bedrock message format
            messages = []
            for msg in history:
                content = msg['content']
                if isinstance(content, str):
                    content = [{"text": content}]
                elif not isinstance(content, list):
                    content = [{"text": str(content)}]
                
                messages.append({
                    "role": msg['role'],
                    "content": content
                })
            
            # Iterate through conversation turns
            for iteration in range(1, self.max_iterations + 1):
                disable_tools = iteration == self.max_iterations
                prompt = "Please provide final answer..." if disable_tools else None
                
                response = await self.invoke_bedrock_model(messages, prompt, disable_tools)
                
                if not response or "<Response>" in response or disable_tools:
                    return response
            
            return response
            
        except Exception as error:
            print(f"Completion error: {error}")
            return f"<Response>An error occurred during completion: {str(error)}</Response>"
    
    def send_message_sync(self, message: str, deep_search: bool = False) -> str:
        """Synchronous message sending (main interface)"""
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Run async completion
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(self.start_completion(self.conversation_history))
            
            # Add response to history
            if response:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
            
            return response or "No response generated."
            
        finally:
            loop.close()
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.session_id = None

# Integration with existing ASR system
class LambdaWebSocketClient:
    """WebSocket client that uses the lambda functionality directly"""
    
    def __init__(self, region: str = "us-east-1", model_id: str = None):
        self.lambda_client = LambdaLLMClient(region=region, model_id=model_id)
        self.conversation_history = []
    
    def send_message_sync(self, message: str, deep_search: bool = False) -> str:
        """Send message using lambda functionality"""
        return self.lambda_client.send_message_sync(message, deep_search)
    
    def clear_history(self):
        """Clear conversation history"""
        self.lambda_client.clear_history()
        self.conversation_history = []

# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Test the lambda LLM client
    client = LambdaLLMClient()
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What is ARC DECRA?"
    
    print(f"Testing query: {query}")
    response = client.send_message_sync(query)
    print(f"Response: {response}")
