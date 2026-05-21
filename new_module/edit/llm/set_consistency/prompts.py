
EDIT_PROMPT = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only a few of the {datapoint_type}(s).
# Requirements
- Edit only some of the {datapoint_type}(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited {datapoint_type}(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the edited {datapoint_type}(s) were changed, and the output is the complete revised text.
# Input
{input_text}
"""
EDIT_WITH_LOCATE_PROMPT = """# Role
You are an expert logician.
# Task
Inspect the provided text and eliminate any logical contradiction by editing only the specified {datapoint_type}(s).
# Requirements
- Edit only the indicated {datapoint_type}(s).
- Resolve the contradiction in the text.
- Preserve the rest of the text unchanged.
- Do not change wording, order, punctuation, or capitalization outside the edited {datapoint_type}(s).
# Output Format
- Return only the fully revised text as plain text.
- Output exactly the revised text and nothing else.
- Do not include explanations or additional formatting.
# Final Check
Before finalizing, verify that the contradiction is resolved, only the allowed {datapoint_type} index was edited, and the output is the complete revised text.
# Input
**Input Text:**
{input_text}
**{datapoint_type_capitalized} Indexes to Edit:**
- {locate_labels}"""
