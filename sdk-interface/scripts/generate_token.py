#!/usr/bin/env python3
"""
Generate a secure API token for bearer authentication.

Usage:
    python scripts/generate_token.py
    make generate-token
"""

import click
import secrets
import hashlib


def generate_secure_token(username: str, length: int = 32) -> str:
    """
    Generate a secure token using secrets module with salted hash.
    
    Args:
        username: Username to include in salt
        length: Length of random bytes to generate (default: 32)
    
    Returns:
        Token in format: op_wui_<hex_string>
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(length)
    
    # Create salt from username
    salt = hashlib.sha256(username.encode()).digest()
    
    # Combine salt and random bytes
    combined = salt + random_bytes
    
    # Hash the combined data
    hashed = hashlib.sha256(combined).hexdigest()
    
    # Take first 32 characters for a reasonable token length
    token_suffix = hashed[:32]
    
    return f"op_wui_{token_suffix}"


@click.command()
@click.option(
    '--username',
    prompt='Enter username for logs',
    help='Username to associate with this token (used for logging purposes)'
)
@click.option(
    '--length',
    default=32,
    help='Length of random bytes to generate (default: 32)'
)
def main(username: str, length: int):
    """Generate a secure API token for bearer authentication."""
    
    # Validate username
    username = username.strip()
    if not username:
        click.echo("Error: Username cannot be empty", err=True)
        raise click.Abort()
    
    if ':' in username or ';' in username:
        click.echo("Error: Username cannot contain ':' or ';' characters", err=True)
        raise click.Abort()
    
    # Generate token
    token = generate_secure_token(username, length)
    
    # Display results
    click.echo()
    click.echo("Add to .env:")
    click.echo(f"API_KEYS={username}:{token}")
    click.echo()
    click.echo("(If API_KEYS exists, append with semicolon)")
    click.echo()


if __name__ == '__main__':
    main()
