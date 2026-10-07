#!/usr/bin/env python3
"""Send one request to the Claude API, authenticating with Workload Identity Federation.

The workflow's OIDC token is read from a file and exchanged for a short-lived Claude API
access token, so no API key is stored. Every setting comes from the environment:

  ANTHROPIC_IDENTITY_TOKEN_FILE  path of the file that holds the OIDC token (required)
  ANTHROPIC_FEDERATION_RULE_ID   fdrl_... (required)
  ANTHROPIC_ORGANIZATION_ID      organization UUID (required)
  ANTHROPIC_SERVICE_ACCOUNT_ID   svac_... (required)
  ANTHROPIC_WORKSPACE_ID         wrkspc_... (optional: needed only when the federation rule
                                 covers more than one workspace)
  CLAUDE_MODEL                   model to call (optional, default claude-sonnet-5-5)
"""

import os
import sys

import anthropic
from anthropic import IdentityTokenFile, WorkloadIdentityCredentials

DEFAULT_MODEL = "claude-sonnet-5-5"
REQUIRED = (
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
)


def main() -> int:
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        print(f"::error::Missing environment variables: {', '.join(missing)}")
        return 1
    token_file = os.environ["ANTHROPIC_IDENTITY_TOKEN_FILE"]
    if not os.path.isfile(token_file) or os.path.getsize(token_file) == 0:
        print(f"::error::The identity token file {token_file} is missing or empty.")
        return 1

    # Passing credentials= makes the client use federation only, even if an API key
    # happens to be set in the environment.
    client = anthropic.Anthropic(
        credentials=WorkloadIdentityCredentials(
            identity_token_provider=IdentityTokenFile(token_file),
            federation_rule_id=os.environ["ANTHROPIC_FEDERATION_RULE_ID"],
            organization_id=os.environ["ANTHROPIC_ORGANIZATION_ID"],
            service_account_id=os.environ["ANTHROPIC_SERVICE_ACCOUNT_ID"],
            # An empty CI variable means "not set".
            workspace_id=os.environ.get("ANTHROPIC_WORKSPACE_ID") or None,
        ),
    )

    model = os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
    try:
        message = client.beta.messages.create(
            model=model,
            max_tokens=1024,
            # If the safety classifiers decline the request, the API retries it on
            # Anthropic's recommended fallback model instead of returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": "Hello, Claude"}],
        )
    except anthropic.WorkloadIdentityError as error:
        print(f"::error::Could not exchange the OIDC token for a Claude API token: {error}")
        return 1
    except anthropic.AuthenticationError as error:
        print(f"::error::The token exchange or request was not authorised: {error.message}")
        return 1
    except anthropic.APIStatusError as error:
        print(f"::error::The Claude API returned HTTP {error.status_code}: {error.message}")
        return 1
    except anthropic.APIConnectionError as error:
        print(f"::error::Could not reach the Claude API: {error}")
        return 1

    if message.stop_reason == "refusal":
        print(f"::error::The request was declined: {message.stop_details}")
        return 1
    text = "".join(block.text for block in message.content if block.type == "text")
    print(f"Model: {message.model}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
