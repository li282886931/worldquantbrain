# Dotenv Credential Loading Design

## Goal

All WorldQuant BRAIN and PostgreSQL usernames and passwords must come from the
project root `.env` file. Values in `.env` take precedence over values already
present in the process environment.

## Design

- Add a small `config.py` module responsible for loading the project `.env`.
- Use `python-dotenv` with `override=True` and validate required values after
  loading.
- Standardize BRAIN credentials on `WQB_USERNAME` and `WQB_PASSWORD`.
- Remove username and password arguments from login functions so callers cannot
  bypass the configured source.
- Make the wqbkit validator load the selected project root `.env` with the same
  override behavior before importing wqbkit.
- Keep Docker Compose interpolation through `${DB_USER}` and `${DB_PASSWORD}`;
  Compose already reads the project `.env` file.
- Update both notebooks to use the shared login path and variable names.

## Errors

Missing or blank required values raise `RuntimeError` naming every missing key.
The error must not include any credential value.

## Tests

- Prove `.env` values override conflicting process environment values.
- Prove missing values fail before an authentication request is sent.
- Prove both standard and HK login paths use the shared loader.
- Prove the validator uses override semantics for its project `.env`.
- Run the complete unit test suite and scan first-party files for old credential
  names and direct credential reads.
