def validate_agent_response(content: str, use_justification: bool = True) -> bool:
    """
    Validate that the agent response has proper scratchpad and justification format.

    Args:
        content: The agent's response content
        use_justification: Whether justification section is required

    Returns:
        True if the content is valid, False otherwise

    Valid format:
    - Must have exactly one outermost <scratchpad>...</scratchpad> section
    - If use_justification=True, must have exactly one outermost <justification>...</justification> section
    - Justification must come after scratchpad (whitespace between sections is allowed)
    - No non-whitespace text outside these sections
    - Nested tags inside sections are allowed (including nested scratchpad/justification tags)
    """
    content = content.strip()
    if not content:
        return False

    # Find scratchpad section
    scratchpad_section = _extract_outermost_section(content, "scratchpad", 0)
    if not scratchpad_section or scratchpad_section[0] != 0:
        return False

    scratchpad_end = scratchpad_section[1]

    if use_justification:
        # Find justification section after scratchpad
        justification_section = _extract_outermost_section(content, "justification", scratchpad_end)
        if not justification_section:
            return False

        justification_start, justification_end = justification_section

        # Check only whitespace between sections and content ends with justification
        between_content = content[scratchpad_end:justification_start]
        return not between_content.strip() and justification_end == len(content)
    else:
        # For scratchpad only, check content ends with scratchpad
        return scratchpad_end == len(content)


def _extract_outermost_section(content: str, tag_name: str, start_pos: int = 0):
    """
    Extract the outermost section of the given tag type starting from start_pos.
    Returns (start_pos, end_pos) of the entire section including tags, or None if not found.
    """
    opening_tag = f"<{tag_name}>"
    closing_tag = f"</{tag_name}>"

    # Find the first opening tag
    tag_start = content.find(opening_tag, start_pos)
    if tag_start == -1:
        return None

    # Use a stack to find the matching closing tag
    stack = 0
    pos = tag_start

    while pos < len(content):
        # Look for the next opening or closing tag
        next_open = content.find(opening_tag, pos)
        next_close = content.find(closing_tag, pos)

        # Determine which comes first
        if next_open == -1 and next_close == -1:
            # No more tags found
            break
        elif next_open == -1:
            # Only closing tag found
            next_pos = next_close
            is_opening = False
        elif next_close == -1:
            # Only opening tag found
            next_pos = next_open
            is_opening = True
        else:
            # Both found, take the earlier one
            if next_open < next_close:
                next_pos = next_open
                is_opening = True
            else:
                next_pos = next_close
                is_opening = False

        if is_opening:
            stack += 1
            pos = next_pos + len(opening_tag)
        else:
            stack -= 1
            pos = next_pos + len(closing_tag)

            # If stack is 0, we found the matching closing tag
            if stack == 0:
                return (tag_start, pos)

    # No matching closing tag found
    return None


def try_fix_justification_tags(content: str) -> str:
    """
    Attempt to fix unclosed justification tags by adding closing tags.

    Args:
        content: The content that may have unclosed justification tags

    Returns:
        The content with potentially fixed justification tags
    """
    content = content.strip()

    # Look for opening justification tags without matching closing tags
    opening_positions = []
    closing_positions = []

    pos = 0
    while True:
        open_pos = content.find("<justification>", pos)
        if open_pos == -1:
            break
        opening_positions.append(open_pos)
        pos = open_pos + 1

    pos = 0
    while True:
        close_pos = content.find("</justification>", pos)
        if close_pos == -1:
            break
        closing_positions.append(close_pos)
        pos = close_pos + 1

    # If we have more opening tags than closing tags, add closing tags at the end
    if len(opening_positions) > len(closing_positions):
        missing_closing_tags = len(opening_positions) - len(closing_positions)
        content += "</justification>" * missing_closing_tags

    return content


def validate_and_fix_agent_response(content: str, use_justification: bool = True) -> tuple[bool, str]:
    """
    Validate agent response and attempt to fix it if invalid.

    Args:
        content: The agent's response content
        use_justification: Whether justification section is required

    Returns:
        Tuple of (is_valid, fixed_content)
    """
    content = content.strip()

    # First, try validation as-is
    if validate_agent_response(content, use_justification):
        return True, content

    # If validation fails and we're using justification, try to fix justification tags
    if use_justification:
        fixed_content = try_fix_justification_tags(content)
        if validate_agent_response(fixed_content, use_justification):
            return True, fixed_content

    # If we still can't fix it, return the original content as invalid
    return False, content
