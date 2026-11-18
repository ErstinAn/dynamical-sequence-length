"""Check the W&B Cloud account and project before a training job starts."""

import argparse
import os

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity")
    parser.add_argument("project")
    args = parser.parse_args()
    query = """
        query CheckAccess($entity: String!, $project: String!) {
            viewer { username }
            project(entityName: $entity, name: $project) { name readOnly }
        }
    """
    key = os.environ.get("WANDB_API_KEY")
    # With no environment key, Requests can use the W&B login in .netrc.
    try:
        response = requests.post(
            "https://api.wandb.ai/graphql",
            auth=("api", key) if key else None,
            json={"query": query, "variables": vars(args)},
            timeout=(5, 15),
            allow_redirects=False,
        )
    except requests.RequestException:
        raise SystemExit("Cannot reach api.wandb.ai. Check DNS, proxy, TLS, and network access.")
    if response.status_code == 401:
        raise SystemExit("W&B rejected the login. Check the key available to this process.")
    if response.status_code == 403:
        raise SystemExit("W&B denied access. Check the account, entity, and project permissions.")
    if response.status_code != 200:
        raise SystemExit(f"W&B returned HTTP {response.status_code}. Check the service and network.")
    try:
        payload = response.json()
    except ValueError:
        raise SystemExit("Expected a JSON response from api.wandb.ai. Check the network or proxy.")
    data = payload.get("data") or {}
    if not (data.get("viewer") or {}).get("username"):
        raise SystemExit("W&B could not verify the account. Check the supplied or saved key.")
    project = data.get("project") or {}
    if payload.get("errors") or project.get("name") != args.project or project.get("readOnly") is not False:
        raise SystemExit(f"Check the exact name and write access for {args.entity}/{args.project}.")
    print(f"W&B login OK. Project permits writes: {args.entity}/{args.project}")


if __name__ == "__main__":
    main()
