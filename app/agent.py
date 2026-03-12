from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from .config import settings

SYSTEM_PROMPT = """You are the ThoughtMinds Permit Operation Assistant — an expert AI system for vendor qualification and permit compliance analysis.

Your responsibilities:
- Guide vendors through uploading permit documents for compliance review
- Interpret and clearly explain compliance analysis results
- Highlight critical deviations, risk scores, and actionable recommendations
- Provide clear next steps based on the compliance decision

Response style:
- Be professional, clear, and concise
- When presenting analysis results, lead with the decision (APPROVE / REJECT / ESCALATE)
- Explain the decision reasoning in plain language
- For rejected submissions, clearly list what needs to be corrected
- Use structured formatting with line breaks for readability"""


def _build_result_context(result: Dict[str, Any], filename: Optional[str]) -> str:
    if not result:
        return ""

    lines = ["--- COMPLIANCE ANALYSIS RESULT ---"]
    if filename:
        lines.append(f"File: {filename}")

    lines += [
        f"Decision: {result.get('decision', 'PENDING')}",
        f"Validation Score: {result.get('validation_score', 0)}%",
        f"Vendor Rating: {result.get('vendor_rating', 0)}/5",
        f"Composite Risk Score: {result.get('composite_risk_score', 0)}",
        f"Decision Reasoning: {result.get('decision_reasoning', 'N/A')}",
    ]

    deviations = result.get("deviations", [])
    if deviations:
        lines.append(f"\nDeviations ({len(deviations)}):")
        for d in deviations:
            cls = d.get("classification", "unknown").upper()
            lines.append(
                f"  [{cls}] {d.get('item', '')}: "
                f"expected '{d.get('expected', '')}', found '{d.get('actual', '')}'"
            )

    recommendations = result.get("recommendations", [])
    if recommendations:
        lines.append("\nRecommendations:")
        for r in recommendations:
            lines.append(f"  - {r}")

    workflow = result.get("workflow", {})
    if workflow.get("tool_executed") and workflow.get("execution_status"):
        lines.append(
            f"\nWorkflow Action: {workflow['tool_executed']} ({workflow['execution_status']})"
        )

    lines.append("--- END RESULT ---")
    return "\n".join(lines)


async def generate_chat_response(
    messages: List[Dict[str, str]],
    workflow_result: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> str:
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    openai_messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if role in ("user", "assistant") and content:
            openai_messages.append({"role": role, "content": content})

    # Ensure there is at least one user message
    if len(openai_messages) == 1:
        openai_messages.append({"role": "user", "content": "Hello"})

    # Inject the workflow result into the last user turn
    if workflow_result is not None:
        context = _build_result_context(workflow_result, filename)
        last = openai_messages[-1]
        if last["role"] == "user":
            existing = last["content"]
            openai_messages[-1] = {
                "role": "user",
                "content": (
                    f"{existing}\n\n{context}"
                    if existing
                    else f"I have uploaded a document for compliance analysis.\n\n{context}"
                ),
            }
        else:
            openai_messages.append({
                "role": "user",
                "content": (
                    f"The document analysis is complete. "
                    f"Please provide a comprehensive response.\n\n{context}"
                ),
            })

    response = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=openai_messages,  # type: ignore[arg-type]
    )

    return response.choices[0].message.content or ""
