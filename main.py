from vllm import LLM, SamplingParams

prompts = [
    "You are an assistant to a Software Engineer. Your task is to assist him with responses limited to Software \
    Engineering practices",
    "Discuss the pros and cons of Agile development in about 500 words",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="adept/persimmon-8b-chat")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")