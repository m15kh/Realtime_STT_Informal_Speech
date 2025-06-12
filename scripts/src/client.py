import asyncio
import websockets
import signal
import sys
from SmartAITool.core import bprint, cprint
SERVER_URL = "ws://localhost:8000/ws"

async def send_text_prompt(websocket):
    """Continuously send text prompts to the server."""
    print("Type your prompts below (Ctrl+C to exit):")
    while True:
        try:
            prompt = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("")
            )
            if prompt.strip():
                bprint(f"Sending prompt:", length=30)
                cprint(f"{prompt}", "green")
                bprint(length=40)
                await websocket.send(prompt)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error sending prompt: {e}")
            break

async def main():
    """Main client function to interact with the server."""
    print(f"Connecting to {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL) as websocket:
            print("Connected to server.")
            
            # Set up tasks for sending prompts and listening for responses
            send_task = asyncio.create_task(send_text_prompt(websocket))
            response_task = asyncio.create_task(listen_for_responses(websocket))
            
            await asyncio.gather(send_task, response_task)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed: {e}")
    except Exception as e:
        print(f"Connection error: {e}")

async def listen_for_responses(websocket):
    """Listen for and process responses from the server."""
    try:
        while True:
            response = await websocket.recv()
            process_response(response)
    except websockets.exceptions.ConnectionClosedError:
        print("Connection to server closed.")
    except Exception as e:
        print(f"Error receiving message: {e}")

def process_response(response):
    """Process different types of responses from the server."""
    if response.startswith("[LLM_RESPONSE_TEXT]"):
        llm_text = response.replace("[LLM_RESPONSE_TEXT]", "").strip()
        bprint("LLM Response",length=30)
        cprint(llm_text, 'blue')
        bprint(length=40)
    elif response.startswith("[ERROR]"):
        error_msg = response.replace("[ERROR]", "").strip()
        print(f"\nError: {error_msg}")
    elif response.startswith("[INFO]"):
        info_msg = response.replace("[INFO]", "").strip()
        print(f"\nInfo: {info_msg}")
    elif response.startswith("[PROMPT]"):
        # Skip printing the prompt echo from server
        pass
    elif response.startswith("[NEXT_PROMPT]"):
        print("\nEnter your prompt: ", end='', flush=True)
    else:
        print(f"Server message: {response}")

def handle_interrupt(signal, frame):
    """Handle keyboard interrupt to gracefully exit."""
    print("\nExiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
