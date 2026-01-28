# Deep Research Feature

## What is Deep Research?

Deep Research is a comprehensive research mode that uses Gemini 2.0 Flash Thinking models to provide extensive, multi-perspective analysis with detailed reasoning.

## How to Use in Open WebUI

### Option 1: Use as a Model (RECOMMENDED)

Simply select the deep research model from your model list:

**Model Name:** `gemini-2.0-flash-thinking-deep-research`

This model appears in your model dropdown alongside other models. When selected:
- Automatically enables deep research mode
- Always streams responses
- Shows extensive reasoning in the thinking panel
- Provides comprehensive, multi-angle analysis

### Option 2: Use via Direct API Endpoint

```bash
POST /v1/deep-research
```

## Features

### Automatic Deep Research Behaviors
When using deep research models:
- ✅ High reasoning effort (extensive thinking)
- ✅ Extended token budget (8000 tokens)
- ✅ Research-focused system prompt
- ✅ Multi-perspective analysis
- ✅ Detailed reasoning steps visible in UI

### What You'll See

**In the Reasoning Panel:**
- `[Deep Research]` meta-messages about process
- `[SDK]` connection status
- Topic breakdown and analysis
- Research methodology
- Multiple perspective exploration
- Synthesis of findings

**In the Main Response:**
- Comprehensive answer
- Well-supported conclusions
- References to reasoning process

## Use Cases

Perfect for:
- 📚 Research questions requiring depth
- 🔍 Complex topics needing multiple angles
- 🧠 Problems requiring step-by-step analysis
- 📊 Topics that benefit from synthesis
- 🎓 Educational deep-dives

## Example Queries

Good deep research prompts:
```
"Research: How does quantum entanglement work and what are its practical applications?"

"Analyze: What are the key differences between various machine learning architectures?"

"Investigate: What factors contributed to the success of electric vehicles?"

"Explore: How do different programming paradigms compare for large-scale systems?"
```

## Configuration

### Reasoning Effort
While deep research defaults to "high" effort, you can override:

```json
{
  "model": "gemini-2.0-flash-thinking-deep-research",
  "messages": [...],
  "reasoning_effort": "medium"  // Override if needed
}
```

### Max Tokens
Default is 8000 for comprehensive responses. Adjust if needed:

```json
{
  "model": "gemini-2.0-flash-thinking-deep-research",
  "messages": [...],
  "max_tokens": 4000  // Shorter response
}
```

## Technical Details

### How It Works
1. Model name detection: `-deep-research` suffix
2. Automatic routing to deep research handler
3. Injection of research-focused system prompt
4. High reasoning effort configuration
5. Streaming with detailed reasoning_content

### Base Model
The deep research feature uses `gemini-2.0-flash-thinking-exp` as its base model.

### Streaming Only
Deep research always streams responses (even if `stream: false` is set) to provide real-time visibility into the research process.

## Comparison

### Regular Thinking Model vs Deep Research

**Regular `gemini-2.0-flash-thinking-exp`:**
- Shows thinking in `<thinking>` tags
- Standard prompting
- You control the prompt
- Configurable reasoning effort

**Deep Research `gemini-2.0-flash-thinking-deep-research`:**
- Enhanced with research methodology
- Specialized system prompt
- Multi-perspective by default
- Always high reasoning effort
- Comprehensive analysis

## Best Practices

1. **Be Specific**: Ask clear research questions
2. **Set Context**: Provide relevant background
3. **Expect Detail**: Responses will be thorough
4. **Review Reasoning**: Check the thinking panel
5. **Adjust Tokens**: Use higher limits for complex topics

## Troubleshooting

### Model Not Appearing
- Restart Open WebUI after adding the SDK
- Check `/v1/models` endpoint lists the model
- Verify Gemini API key is configured

### No Reasoning Shown
- Check that Open WebUI's reasoning UI is enabled
- Verify model name includes `-deep-research`
- Look for `reasoning_content` in stream chunks

### Incomplete Responses
- Increase `max_tokens` (default: 8000)
- Check API logs for errors
- Verify network connectivity

## Future Enhancements

Planned features:
- [ ] Multiple deep research variants (quick, thorough, exhaustive)
- [ ] Custom research templates
- [ ] Source citation integration
- [ ] Multi-model research synthesis
- [ ] Research report generation
