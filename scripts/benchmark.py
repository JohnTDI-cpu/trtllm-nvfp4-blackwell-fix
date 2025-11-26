import time
import requests
import argparse
import sys

def benchmark(url, model, prompt_len=128, max_tokens=512, stream=False):
    print(f"🚀 Benchmarking Model: {model}")
    print(f"🎯 Target: {url}")
    print(f"⚙️  Config: prompt_len={prompt_len}, max_tokens={max_tokens}, stream={stream}")
    
    # Generate a dummy prompt of approx length
    prompt = "word " * prompt_len
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.1, # Low temp for consistency
        "top_p": 0.9
    }
    
    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        if stream:
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        if first_token_time is None:
                            first_token_time = time.time()
                        # Simple approximation of tokens based on chunks (not perfect for SSE)
                        token_count += 1 
            end_time = time.time()
            # Stream token count is tricky without parsing, better use non-stream for throughput
            print("\n⚠️  Streaming benchmark is approximate for TTFT only.")
        else:
            r = requests.post(url, json=payload)
            r.raise_for_status()
            end_time = time.time()
            first_token_time = end_time # Approx for non-stream
            
            data = r.json()
            if 'usage' in data:
                token_count = data['usage']['completion_tokens']
                prompt_tokens = data['usage']['prompt_tokens']
            else:
                print("Error: No usage data in response")
                return

    except Exception as e:
        print(f"\n❌ Benchmark Failed: {e}")
        return

    total_time = end_time - start_time
    ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
    throughput = token_count / total_time if total_time > 0 else 0

    print("\n📊 Results:")
    print(f"------------------------------------------------")
    print(f"✅ Status:          Success")
    print(f"⏱️  Total Time:      {total_time:.4f} s")
    print(f"⚡ TTFT (Latency):  {ttft:.2f} ms")
    print(f"📦 Tokens Gen:      {token_count}")
    print(f"🚀 Throughput:      {throughput:.2f} tokens/s")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple HTTP Benchmark for LLM")
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions", help="API Endpoint")
    parser.add_argument("--model", required=True, help="Model name (ID)")
    parser.add_argument("--tokens", type=int, default=512, help="Max tokens to generate")
    args = parser.parse_args()
    
    benchmark(args.url, args.model, max_tokens=args.tokens)

