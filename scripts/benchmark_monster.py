import time
import openai
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown

# Configuration
API_BASE = "http://localhost:8000/v1"
API_KEY = "EMPTY"

console = Console()

def get_active_model(client):
    try:
        models = client.models.list()
        # Assuming the first model is the correct one
        model_id = models.data[0].id
        console.print(f"[bold blue]ℹ️ Model detected:[/bold blue] [green]{model_id}[/green]")
        return model_id
    except Exception as e:
        console.print(f"[bold red]❌ Error retrieving model list: {e}[/bold red]")
        return "Qwen3-30B-NVFP4-Engine"

def run_benchmark(client, model_name, prompt, label, max_tokens=1000, temperature=0.1):
    console.print(f"\n[bold yellow]▶ Test: {label}[/bold yellow]")
    
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    full_response = ""
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                content = chunk.choices[0].delta.content
                token_count += 1
                full_response += content
                # Optional: print a dot for each token
                # console.print(".", end="", style="dim")

        end_time = time.perf_counter()

        if first_token_time is None:
            console.print("[bold red]❌ ERROR: No response (empty stream)![/bold red]")
            return 0, 0, 0, ""

        ttft = (first_token_time - start_time) * 1000 # ms
        total_gen_time = end_time - first_token_time
        tps = token_count / total_gen_time if total_gen_time > 0 else 0
        
        return ttft, tps, token_count, full_response

    except Exception as e:
        console.print(f"[bold red]❌ CRITICAL ERROR: {e}[/bold red]")
        return 0, 0, 0, ""

def main():
    client = openai.OpenAI(base_url=API_BASE, api_key=API_KEY)
    console.print(Panel.fit("[bold cyan]RTX 5090 NVFP4 - PERFORMANCE BENCHMARK[/bold cyan]\n[dim]Target: Qwen 30B MoE (NVFP4)[/dim]"))

    MODEL_NAME = get_active_model(client)

    table = Table(title="Benchmark Results")
    table.add_column("Test Case", style="white")
    table.add_column("TTFT (ms)", style="white")
    table.add_column("Throughput (tokens/s)", style="green", justify="right")
    table.add_column("Total Tokens", style="white")
    table.add_column("Result", style="bold")

    # 1. WARMUP
    console.print("[dim]Initializing engine (Warmup)...[/dim]")
    run_benchmark(client, MODEL_NAME, "Hi", "Warmup", max_tokens=10)

    # 2. LOGIC TEST (Sanity Check)
    logic_prompt = "I have 3 apples. I ate one, then bought two more. How many apples do I have? Explain step by step."
    ttft, tps, tokens, text = run_benchmark(client, MODEL_NAME, logic_prompt, "Logic/Reasoning Test", max_tokens=200)
    
    # Check if response is sane (looking for "4" or "four")
    is_sane = "4" in text or "four" in text.lower()
    status_icon = "PASS" if is_sane else "FAIL"
    status_style = "green" if is_sane else "red"
    table.add_row("Logic Check", f"{ttft:.1f} ms", f"{tps:.1f}", str(tokens), f"[{status_style}]{status_icon}[/{status_style}]")
    
    console.print(Panel(Markdown(f"**Model Response:**\n{text}"), title="Logic Output", border_style="white"))

    # 3. SPEED TEST (Long Generation)
    speed_prompt = "Write a very long, detailed technical essay explaining the architecture of Mixture of Experts (MoE) models and how 4-bit quantization (NVFP4) impacts their memory bandwidth usage. Write at least 1000 words."
    # Reduced max_tokens to fit within 2048 limit (approx 2048 - prompt_len)
    ttft, tps, tokens, text = run_benchmark(client, MODEL_NAME, speed_prompt, "Throughput Test (Long Context)", max_tokens=1800)
    table.add_row("Max Throughput", f"{ttft:.1f} ms", f"[bold]{tps:.1f}[/bold]", str(tokens), "[green]COMPLETED[/green]")

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]Configuration:[/bold] NVFP4 (4-bit) | TensorRT-LLM v1.2.0rc4 | CUDA 13.0")

if __name__ == "__main__":
    main()
