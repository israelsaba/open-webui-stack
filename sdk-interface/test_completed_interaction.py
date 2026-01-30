import requests
import json

url = 'http://localhost:8000/v1/chat/completions'
headers = {'Authorization': 'Bearer op_wui_e40491f279be12e272754112d5c0228a'}

# This message hash should match the completed interaction
payload = {
    'model': 'deep-research-pro-preview-12-2025',
    'messages': [{'role': 'user', 'content': 'write a poem about software developers'}],
    'stream': True
}

print('Testing retrieval of completed interaction...')
print('Expected interaction_id: v1_ChdSNnA4YWIyRkxKV2N6N0lQMmYzbmtBURIXUjZwOGFiMkZMSldjejdJUDJmM25rQVE')
print()

response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)

has_content = False
has_reasoning = False
content_length = 0

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = line[6:].decode('utf-8')
        if data == '[DONE]':
            print('\n[DONE]')
            break
        try:
            chunk = json.loads(data)
            delta = chunk['choices'][0].get('delta', {})
            if 'reasoning_content' in delta:
                has_reasoning = True
                content = delta['reasoning_content']
                if 'Continuing interaction with id' in content:
                    print(f'✓ Found continuation message')
                elif '[SDK]' in content:
                    print(f'[SDK] {content.strip()}')
            if 'content' in delta:
                has_content = True
                content_length += len(delta['content'])
                if content_length < 200:
                    print(delta['content'], end='', flush=True)
        except Exception as e:
            print(f'Error parsing: {e}')

print(f'\n\n=== Results ===')
print(f'Has reasoning: {has_reasoning}')
print(f'Has content: {has_content}')
print(f'Content length: {content_length}')

if has_content and content_length > 100:
    print('\n✅ SUCCESS: Completed interaction output was retrieved!')
else:
    print('\n❌ FAILED: No content output from completed interaction')
