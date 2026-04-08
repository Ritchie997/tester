#!/usr/bin/env python3
"""
Production-ready CLI script for validating large .po files (65k+ entries, EN→RU)
using local LLM via Ollama.

Features:
- Two-stage validation: fast regex checks + semantic LLM analysis
- Checkpointing with resume capability
- Retry logic with exponential backoff
- Progress tracking with tqdm
- Outputs: validated .po file with #freez flags + JSONL issues report
"""

import argparse
import json
import logging
import re
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import polib
import requests
from tqdm import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]

RESUME_STATE_FILE = "resume_state.json"
DEFAULT_OUTPUT_PO = "validated_output.po"
DEFAULT_ISSUES_JSONL = "issues.jsonl"
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N entries

# Regex patterns for structural validation
PATTERNS = {
    "placeholder_braces": re.compile(r"\{[^}]*\}"),  # {name}, {0}, etc.
    "placeholder_percent": re.compile(r"%[sdfioxXeEfFgGcr]"),  # %s, %d, etc.
    "newline": re.compile(r"\\n"),  # \n literal
    "html_tag": re.compile(r"<[^>]+>"),  # <tag>, </tag>, etc.
}

SYSTEM_PROMPT = """You are a strict Translation QA Validator for game localization (EN->RU).
Your goal is to find CRITICAL semantic errors that break the game experience or mislead the player.
Output ONLY valid JSON. No markdown, no explanations.

=== CRITICAL ERRORS (FLAG THESE) ===
1. WRONG MEANING: Target says the opposite or something factually different (e.g., "you die" vs "you might die", "safe" -> "dangerous").
2. MISSING CRITICAL INFO: Omission of key gameplay instructions, numbers, conditions, item names, or key mechanics.
3. HALLUCINATION: Target adds info NOT present in source (e.g., specific items, mechanics, or facts not mentioned).
4. BROKEN PLACEHOLDERS: Missing or changed code variables like <press_key>, {variables}, %s (caught by structural check, but flag if semantic context breaks).

=== IGNORE THESE (DO NOT FLAG) ===
1. Stylistic differences (word choice, sentence structure) if meaning is preserved.
2. Shortening long titles (e.g., "Cataclysm: The Last Generation" -> "Катаклизм") if context is clear.
3. Minor omissions of filler words ("simply", "actually", "sort of", "often").
4. Splitting/merging sentences if logical flow is intact (sentence count checked separately).
5. Synonyms that convey the same game mechanic accurately.
6. Generalization that doesn't lose critical info (e.g., "firearms and tools" -> "оружие и инструменты").

=== OUTPUT FORMAT ===
Return ONLY this JSON:
{"is_correct": boolean, "reason": "string or null"}
- If correct (or minor issues only): {"is_correct": true, "reason": null}
- If incorrect (critical error): {"is_correct": false, "reason": "One short sentence stating WHAT is wrong: 'Opposite meaning: X', 'Missing critical: Y', 'Hallucination: Z'"}
"""

USER_PROMPT_TEMPLATE = """Analyze this game localization pair for CRITICAL errors only.
SOURCE: {msgid}
TARGET: {msgstr}

Apply the strict rules above. Ignore stylistic differences. Flag only game-breaking errors.
Return ONLY JSON:"""


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


# =============================================================================
# STRUCTURAL VALIDATION (STAGE 1)
# =============================================================================

def count_sentences(text: str) -> int:
    """Count sentences in text based on terminal punctuation."""
    if not text.strip():
        return 0
    # Split on sentence-ending punctuation followed by space, newline, or end of string
    # Use a more robust pattern that handles multiple punctuation marks
    sentences = re.split(r"[.!?]+\s*", text.strip())
    # Filter out empty strings
    return len([s for s in sentences if s.strip()])


def count_lines(text: str) -> int:
    """Count non-empty lines in text."""
    if not text.strip():
        return 0
    return len([line for line in text.split("\n") if line.strip()])


def extract_placeholders(text: str) -> set:
    """Extract all placeholders from text."""
    placeholders = set()
    
    # Extract {.*} placeholders
    for match in PATTERNS["placeholder_braces"].finditer(text):
        placeholders.add(("braces", match.group()))
    
    # Extract % placeholders
    for match in PATTERNS["placeholder_percent"].finditer(text):
        placeholders.add(("percent", match.group()))
    
    # Extract \n literals
    for match in PATTERNS["newline"].finditer(text):
        placeholders.add(("newline", match.group()))
    
    # Extract HTML tags
    for match in PATTERNS["html_tag"].finditer(text):
        placeholders.add(("html", match.group()))
    
    return placeholders


def validate_structure(msgid: str, msgstr: str) -> tuple[bool, Optional[str]]:
    """
    Stage 1: Fast structural validation.
    Returns (is_valid, error_reason).
    """
    # Check placeholder consistency
    src_placeholders = extract_placeholders(msgid)
    tgt_placeholders = extract_placeholders(msgstr)
    
    if src_placeholders != tgt_placeholders:
        src_only = src_placeholders - tgt_placeholders
        tgt_only = tgt_placeholders - src_placeholders
        reason_parts = []
        if src_only:
            reason_parts.append(f"missing in target: {[p[1] for p in src_only]}")
        if tgt_only:
            reason_parts.append(f"extra in target: {[p[1] for p in tgt_only]}")
        return False, f"placeholder_mismatch: {'; '.join(reason_parts)}"
    
    # Check sentence count (allow small tolerance)
    src_sentences = count_sentences(msgid)
    tgt_sentences = count_sentences(msgstr)
    
    # Allow ±1 sentence difference for edge cases
    if abs(src_sentences - tgt_sentences) > 1:
        return False, f"sentence_count_mismatch: source={src_sentences}, target={tgt_sentences}"
    
    # Check line count (should match exactly for formatted strings)
    src_lines = count_lines(msgid)
    tgt_lines = count_lines(msgstr)
    
    if src_lines != tgt_lines:
        return False, f"line_count_mismatch: source={src_lines}, target={tgt_lines}"
    
    return True, None


# =============================================================================
# LLM VALIDATION (STAGE 2)
# =============================================================================

def call_ollama(msgid: str, msgstr: str, model: str, timeout: int) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Call Ollama API for semantic validation.
    Returns (is_valid, reason, error_message).
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(msgid=msgid, msgstr=msgstr)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": 0.0,  # Deterministic output
            "top_p": 0.1,
        },
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        # Parse JSON response
        try:
            # Try to extract JSON from response (handle potential markdown)
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = content
            
            parsed = json.loads(json_str)
            
            is_correct = parsed.get("is_correct", False)
            reason = parsed.get("reason")
            
            if is_correct:
                return True, None, None
            else:
                return False, reason or "semantic_mismatch", None
                
        except json.JSONDecodeError as e:
            return False, None, f"llm_parse_error: {str(e)}"
            
    except requests.exceptions.Timeout:
        return False, None, "ollama_timeout"
    except requests.exceptions.ConnectionError as e:
        return False, None, f"ollama_connection_error: {str(e)}"
    except requests.exceptions.RequestException as e:
        return False, None, f"ollama_request_error: {str(e)}"


def call_ollama_with_retry(msgid: str, msgstr: str, model: str, timeout: int, logger: logging.Logger) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Call Ollama with retry logic (exponential backoff).
    """
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        is_valid, reason, error = call_ollama(msgid, msgstr, model, timeout)
        
        # If we got a valid response (even if translation is wrong), return it
        if error is None:
            return is_valid, reason, None
        
        # If error is not retryable, return immediately
        if "parse_error" in error:
            return False, reason or error, None
        
        last_error = error
        logger.warning(f"Ollama request failed (attempt {attempt + 1}/{MAX_RETRIES}): {error}")
        
        if attempt < MAX_RETRIES - 1:
            sleep_time = RETRY_DELAYS[attempt]
            logger.info(f"Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
    
    return False, None, f"max_retries_exceeded: {last_error}"


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def load_resume_state(resume_file: str) -> dict:
    """Load resume state from file."""
    path = Path(resume_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": -1, "issues": [], "processed_count": 0, "error_count": 0, "total_to_process": None}


def save_resume_state(state: dict, resume_file: str) -> None:
    """Save resume state to file."""
    with open(Path(resume_file), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def clear_resume_state(resume_file: str) -> None:
    """Remove resume state file."""
    path = Path(resume_file)
    if path.exists():
        path.unlink()


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logging.info("\nInterrupt signal received. Will save checkpoint after current entry...")


# =============================================================================
# PO FILE PROCESSING
# =============================================================================

def get_entry_text(entry: polib.POEntry) -> str:
    """Get full text from PO entry (handles msgstr plural forms)."""
    if entry.msgstr_plural:
        # Join all plural forms for validation
        return "\n".join(entry.msgstr_plural.values())
    return entry.msgstr or ""


def add_freez_flag(entry: polib.POEntry) -> None:
    """Add #freez flag to PO entry comments."""
    if "freez" not in entry.comment:
        if entry.comment:
            entry.comment = f"#freez\n{entry.comment}"
        else:
            entry.comment = "#freez"


def process_po_file(
    input_path: str,
    output_path: str,
    issues_path: str,
    resume_path: str,
    model: str,
    timeout: int,
    resume: bool,
    limit: Optional[int],
    logger: logging.Logger
) -> None:
    """Main processing function."""
    
    # Load PO file
    logger.info(f"Loading PO file: {input_path}")
    po_file = polib.pofile(input_path, encoding="utf-8")
    total_entries = len(po_file)
    logger.info(f"Total entries in file: {total_entries}")
    
    # Determine how many entries to process
    if limit is not None:
        entries_to_process = min(limit, total_entries)
        logger.info(f"Processing up to {entries_to_process} entries (limit specified)")
    else:
        entries_to_process = total_entries
        logger.info(f"Processing all {total_entries} entries")
    
    # Initialize or load resume state
    if resume:
        state = load_resume_state(resume_path)
        start_index = state["last_index"] + 1
        issues = state["issues"]
        processed_count = state["processed_count"]
        error_count = state["error_count"]
        # Check if limit changed from previous run
        prev_limit = state.get("total_to_process")
        if prev_limit is not None and limit != prev_limit:
            logger.info(f"Limit changed from {prev_limit} to {limit}. Continuing from index {start_index}.")
        logger.info(f"Resuming from index {start_index} (previously processed: {processed_count})")
    else:
        state = {"last_index": -1, "issues": [], "processed_count": 0, "error_count": 0, "total_to_process": limit}
        start_index = 0
        issues = []
        processed_count = 0
        error_count = 0
        # Clear any existing resume file
        clear_resume_state(resume_path)
    
    # Calculate end index based on limit
    end_index = min(start_index + (entries_to_process - start_index), total_entries)
    if limit is not None and not resume:
        end_index = min(limit, total_entries)
    elif limit is not None and resume:
        # When resuming with a new limit, recalculate
        remaining = limit - processed_count
        end_index = min(start_index + remaining, total_entries)
    
    logger.info(f"Will process entries from index {start_index} to {end_index - 1}")
    
    # Track which entries need #freez flag (for post-processing)
    freez_indices = {issue["index"] for issue in issues}
    
    # Process entries with progress bar
    with tqdm(total=end_index, initial=start_index, desc="Validating", unit="entry") as pbar:
        for idx in range(start_index, end_index):
            entry = po_file[idx]
            
            # Skip entries without msgid or msgstr
            if not entry.msgid or not get_entry_text(entry).strip():
                state["last_index"] = idx
                processed_count += 1
                pbar.update(1)
                continue
            
            msgid = entry.msgid
            msgstr = get_entry_text(entry)
            
            # Stage 1: Structural validation
            struct_valid, struct_error = validate_structure(msgid, msgstr)
            
            if not struct_valid:
                # Structural issue - mark as frozen, skip LLM
                issue = {
                    "index": idx,
                    "msgid": msgid[:500],  # Truncate for JSONL
                    "msgstr": msgstr[:500],
                    "reason": struct_error,
                    "type": "structural"
                }
                issues.append(issue)
                freez_indices.add(idx)
                error_count += 1
                logger.debug(f"Index {idx}: Structural issue - {struct_error}")
                
            else:
                # Stage 2: Semantic validation via LLM
                is_valid, reason, error = call_ollama_with_retry(msgid, msgstr, model, timeout, logger)
                
                if error:
                    # LLM error - treat as issue
                    issue = {
                        "index": idx,
                        "msgid": msgid[:500],
                        "msgstr": msgstr[:500],
                        "reason": error,
                        "type": "semantic"
                    }
                    issues.append(issue)
                    freez_indices.add(idx)
                    error_count += 1
                    logger.debug(f"Index {idx}: LLM error - {error}")
                elif not is_valid:
                    # Semantic issue
                    issue = {
                        "index": idx,
                        "msgid": msgid[:500],
                        "msgstr": msgstr[:500],
                        "reason": reason or "semantic_mismatch",
                        "type": "semantic"
                    }
                    issues.append(issue)
                    freez_indices.add(idx)
                    error_count += 1
                    logger.debug(f"Index {idx}: Semantic issue - {reason}")
            
            # Update state
            state["last_index"] = idx
            state["issues"] = issues
            state["processed_count"] = processed_count + 1
            state["error_count"] = error_count
            state["total_to_process"] = limit
            
            # Save checkpoint every N entries or on shutdown request
            if (idx + 1) % CHECKPOINT_INTERVAL == 0 or shutdown_requested:
                save_resume_state(state, resume_path)
                if shutdown_requested:
                    logger.info(f"Checkpoint saved at index {idx}. Stopping gracefully.")
                    break
            
            processed_count += 1
            pbar.update(1)
            
            # Update progress bar description with stats
            elapsed = float(pbar.format_dict["elapsed"]) or 0.1
            pbar.set_postfix({"errors": error_count, "rate": f"{processed_count/elapsed:.1f}/s"})
        
        # Check if we stopped due to shutdown
        if shutdown_requested:
            logger.info("Graceful stop completed. Use --resume to continue.")
    
    # Apply #freez flags to PO file
    logger.info("Applying #freez flags to problematic entries...")
    for idx in freez_indices:
        add_freez_flag(po_file[idx])
    
    # Save validated PO file
    logger.info(f"Saving validated PO file: {output_path}")
    po_file.save(output_path)
    
    # Save issues as JSONL
    logger.info(f"Saving issues report: {issues_path}")
    with open(issues_path, "w", encoding="utf-8") as f:
        for issue in issues:
            f.write(json.dumps(issue, ensure_ascii=False) + "\n")
    
    # Clear resume state on successful completion
    clear_resume_state(resume_path)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("VALIDATION COMPLETE")
    logger.info(f"Total entries processed: {processed_count}")
    logger.fatal(f"Issues found: {len(issues)}")
    logger.info(f"  - Structural: {sum(1 for i in issues if i['type'] == 'structural')}")
    logger.info(f"  - Semantic: {sum(1 for i in issues if i['type'] == 'semantic')}")
    logger.info(f"Output files:")
    logger.info(f"  - Validated PO: {output_path}")
    logger.info(f"  - Issues JSONL: {issues_path}")
    logger.info("=" * 60)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate .po files using structural checks and LLM semantic analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.po                          # Full validation
  %(prog)s input.po --resume                 # Resume from checkpoint
  %(prog)s input.po --model mistral          # Use different model
  %(prog)s input.po --timeout 60             # Increase timeout
        """
    )
    
    parser.add_argument(
        "input",
        help="Path to input .po file"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_PO,
        help=f"Path to output validated .po file (default: {DEFAULT_OUTPUT_PO})"
    )
    parser.add_argument(
        "--issues",
        default=DEFAULT_ISSUES_JSONL,
        help=f"Path to output issues JSONL file (default: {DEFAULT_ISSUES_JSONL})"
    )
    parser.add_argument(
        "--resume-state",
        default=RESUME_STATE_FILE,
        help=f"Path to resume state file (default: {RESUME_STATE_FILE})"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to process (e.g., --limit 1000 for first 1000 entries)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.suffix.lower() == ".po":
        logger.warning(f"Input file does not have .po extension: {input_path}")
    
    # Check Ollama availability (unless resuming)
    if not args.resume:
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if args.model not in models:
                logger.warning(f"Model '{args.model}' not found in Ollama. Available: {models}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Is it running?")
            sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not verify Ollama connection: {e}")
    
    # Run validation
    try:
        process_po_file(
            input_path=str(input_path),
            output_path=args.output,
            issues_path=args.issues,
            resume_path=args.resume_state,
            model=args.model,
            timeout=args.timeout,
            resume=args.resume,
            limit=args.limit,
            logger=logger
        )
    except Exception as e:
        logger.exception(f"Unexpected error during validation: {e}")
        # Save state on unexpected error
        if 'state' in locals():
            save_resume_state(state, args.resume_state)
            logger.info(f"State saved to {args.resume_state} for recovery")
        sys.exit(1)


if __name__ == "__main__":
    main()
