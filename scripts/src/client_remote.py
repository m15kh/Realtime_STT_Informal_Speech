import asyncio
import websockets
import signal
import sys
import argparse
import socket
import yaml
import os
from SmartAITool.core import bprint, cprint

# Path to configuration file
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "/home/fteam6/m15kh/Realtime_STT_Informal_Speech/config.yaml")

# Default connection settings (used if config file is not found)
DEFAULT_HOST = "192.168.1.100"  # Default local network address
DEFAULT_PORT = 8000  # WebSocket port on the server
DEFAULT_TIMEOUT = 10  # Connection timeout in seconds

def load_config():
    """Load configuration from YAML file if it exists, otherwise use defaults."""
    config = {
        "server": {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "timeout": DEFAULT_TIMEOUT
        }
    }
    
    try:
        if os.path.exists(CONFIG_FILE_PATH):
            with open(CONFIG_FILE_PATH, 'r') as file:
                yaml_config = yaml.safe_load(file)
                if yaml_config and isinstance(yaml_config, dict):
                    # Update config with values from YAML
                    if 'server' in yaml_config and isinstance(yaml_config['server'], dict):
                        config['server'].update(yaml_config['server'])
                    print(f"Configuration loaded from {CONFIG_FILE_PATH}")
                else:
                    print(f"Warning: Invalid config format in {CONFIG_FILE_PATH}")
        else:
            print(f"Config file not found at {CONFIG_FILE_PATH}, using defaults")
    except Exception as e:
        print(f"Error loading config: {e}")
    
    return config

def parse_arguments(config):
    """Parse command line arguments with config values as defaults."""
    parser = argparse.ArgumentParser(description='Remote client for Realtime STT Informal Speech')
    parser.add_argument('--host', type=str, default=config['server']['host'],
                        help=f'Server hostname or IP address (default: {config["server"]["host"]})')
    parser.add_argument('--port', type=int, default=config['server']['port'],
                        help=f'Server port (default: {config["server"]["port"]})')
    parser.add_argument('--timeout', type=int, default=config['server']['timeout'],
                        help=f'Connection timeout in seconds (default: {config["server"]["timeout"]})')
    return parser.parse_args()

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
    """Main client function to interact with the remote server."""
    args = parse_arguments()
    SERVER_URL = f"ws://{args.host}:{args.port}/ws"
    
    # Try to resolve hostname to check if it's valid
    try:
        ip_address = socket.gethostbyname(args.host)
        print(f"Resolved {args.host} to IP: {ip_address}")
    except socket.gaierror:
        print(f"Warning: Could not resolve hostname '{args.host}'. Check if the hostname is correct.")
    
    print(f"Connecting to remote server at {SERVER_URL}...")
    try:
        async with websockets.connect(SERVER_URL, ping_interval=20, ping_timeout=20, 
                                     close_timeout=args.timeout) as websocket:
            cprint(f"✓ Connected to server at {args.host}:{args.port}", "green")
            print(f"Connection established with WebSocket ID: {id(websocket)}")
            
            # Set up tasks for sending prompts and listening for responses
            send_task = asyncio.create_task(send_text_prompt(websocket))
            response_task = asyncio.create_task(listen_for_responses(websocket))
            
            await asyncio.gather(send_task, response_task)
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Connection closed by server: {e}")
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Server returned invalid status code: {e}")
        print("This usually means the server is running but the WebSocket endpoint is incorrect.")
    except (ConnectionRefusedError, OSError) as e:
        print(f"Connection refused: {e}")
        print(f"Make sure the server is running on {args.host} and port {args.port} is open.")
        print("Check your firewall settings to ensure the port is accessible.")
    except asyncio.TimeoutError:
        print(f"Connection timed out after {args.timeout} seconds.")
        print("The server might be too slow to respond or unreachable.")
    except Exception as e:
        print(f"Connection error: {e}")
        print(f"Could not connect to {SERVER_URL}")

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
        bprint("LLM Response", length=30)
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
    elif response.startswith("[VOICE_START]"):
        cprint("\nVoice activity detected on server", "yellow")
    elif response.startswith("[VOICE_END]"):
        cprint("\nVoice segment processing complete", "yellow")
    elif response.startswith("[TRANSCRIPTION]"):
        transcription = response.replace("[TRANSCRIPTION]", "").strip()
        bprint("Transcription", length=30)
        cprint(transcription, 'magenta')
        bprint(length=40)
    elif response.startswith("[TTS_DONE]"):
        print("\nText-to-speech generation complete")
    else:
        print(f"Server message: {response}")

def handle_interrupt(signal, frame):
    """Handle keyboard interrupt to gracefully exit."""
    print("\nExiting client...")
    sys.exit(0)

def check_network_connectivity(host, port):
    """Check if the target host and port are reachable."""
    try:
        socket.create_connection((host, port), timeout=5)
        return True
    except (socket.timeout, socket.error):
        return False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Load config and parse args
    config = load_config()
    args = parse_arguments(config)
    
    # Print client information
    print(f"Remote STT Client starting...")
    print(f"Target server: {args.host}:{args.port}")
    
    # Check basic network connectivity
    if check_network_connectivity(args.host, args.port):
        print(f"✓ Network connectivity test passed - {args.host}:{args.port} is reachable")
    else:
        print(f"✗ Network connectivity test failed - {args.host}:{args.port} is unreachable")
        print("Possible reasons:")
        print("1. The server is not running")
        print("2. A firewall is blocking the connection")
        print("3. The hostname or port is incorrect")
        print("\nAttempting WebSocket connection anyway...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify the server is running: ssh into the server and check with 'ps aux | grep server.py'")
        print("2. Check server logs for errors")
        print("3. Verify the port is open: use 'netstat -tulpn | grep 8000' on the server")
        print("4. Check firewall rules: 'sudo iptables -L' on the server")
        print("5. Try connecting from the server to itself as a test")
        print("\nUsage example:")
        print(f"  python client_remote.py --host {args.host} --port {args.port}")
