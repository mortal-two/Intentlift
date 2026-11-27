"""Official MCP stdio server forwarding only the shared compiler tools."""
import asyncio
import json
import os
import sys
from .broker import rpc
from ..core.errors import LiftError


async def serve_async(dispatch):
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError as exc:
        raise LiftError('configuration_failure', 'install intentlift[mcp] for MCP and CLI backends') from exc
    tools = dispatch({'method': 'list'})
    context = dispatch({'method': 'context'})

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name=tool['name'], description=tool['description'],
                    inputSchema=tool['parameters'], annotations=types.ToolAnnotations(
                    readOnlyHint=tool['name'] in {'task_context', 'inspect', 'read_file'},
                    destructiveHint=False, openWorldHint=False)) for tool in tools])

    async def call_tool(ctx, params):
        try:
            result = await asyncio.to_thread(dispatch, {'method': 'call', 'name': params.name,
                                                        'arguments': params.arguments or {}})
        except LiftError as exc:
            result = exc.record()
        return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(result, allow_nan=False))],
                                    structuredContent=result, isError=not result.get('ok', False))

    schemas = {tool['name']: tool['parameters'] for tool in tools}
    server = Server('intentlift', instructions=context['instructions'], on_list_tools=list_tools,
                    on_call_tool=call_tool, get_tool_input_schema=lambda name: schemas.get(name))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def serve(dispatch):
    asyncio.run(serve_async(dispatch))


def main():
    path = os.environ.get('LIFT_AGENT_SOCKET')
    if not path:
        from .mcp import standalone
        return standalone()
    serve(lambda request: rpc(path, request))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
